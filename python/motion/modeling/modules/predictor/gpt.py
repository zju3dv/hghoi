import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from motion.modeling.operators.transformer import (
    TransformerEncoderLayer,
)
from motion.modeling.operators.mlp import MLPBlock, Conv1dBlock
from motion.modeling.operators.embedding import EmbeddingLayer
from motion.modeling.operators.positional_encoding import PositionalEncoding
from motion.modeling.modules.builder import MODULES
from motion.utils.utils import to_device


@MODULES.register_module()
class CausalGPTGoalPose(nn.Module):
    def __init__(
        self,
        action_dim: int = 5,
        env_dim: int = 315,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        pred_keys: list = ["contact"],
        quant_n: int = 512,
        activation: str = "ELU",
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.env_dim = env_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.pred_keys = pred_keys
        self.pred_n = len(pred_keys)
        self.quant_n = quant_n
        self.activation = activation
        self._build_model()

    def _build_model(self):
        self._build_encoder_network()
        self._build_transformer()
        self._build_final_layer()
        self._build_mask()

    def _build_encoder_network(self):
        self.action_encoder = nn.Linear(self.action_dim, self.latent_dim)
        self.env_encoder = nn.Sequential(
            MLPBlock(
                self.env_dim, 256, dropout=self.dropout, activation=self.activation
            ),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
            MLPBlock(
                256, self.latent_dim, dropout=self.dropout, activation=self.activation
            ),
        )

    def _build_transformer(self):
        for k in self.pred_keys:
            setattr(self, f"embed_{k}", EmbeddingLayer(self.quant_n, self.latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(self.latent_dim, self.dropout)
        seq_trans_encoder_layer = TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=self.num_layers
        )

    def _build_mask(self):
        mask = torch.tril(torch.ones(self.pred_n, self.pred_n))
        cond_mask = torch.ones(self.pred_n, 2)
        mask = torch.cat((cond_mask, mask), dim=1)
        cond_mask = torch.zeros(2, self.pred_n + 2)
        cond_mask[0, 0] = 1.0
        cond_mask[0, 1] = 1.0
        cond_mask[1, 0] = 1.0
        cond_mask[1, 1] = 1.0
        mask = torch.cat((cond_mask, mask), dim=0)
        self.register_buffer(
            "mask", ~mask.view(self.pred_n + 2, self.pred_n + 2).bool()
        )

    def _build_final_layer(self):
        for k in self.pred_keys:
            final_layer = nn.Linear(self.latent_dim, self.quant_n)
            setattr(self, f"pred_{k}", final_layer)

    def forward(self, env, action, code, **kwargs):
        """_summary_

        Args:
            env (tensor): [B, C]
            action (tensor): [B, C]
            code (dict(tensor)): [B]

        Returns:
            dict(tensor):
                logits: [B, P, c]
        """
        output = {}
        if not self.training:
            preds, probs = self.predict_code(env, action)
            output["preds"] = preds
            output["probs"] = probs
        env = self.env_encoder(env)[None]
        action = self.action_encoder(action)[None]
        condition = torch.cat((env, action), dim=0)  # (2, B, C)

        embeds = []
        for k in self.pred_keys:
            embed_layer = getattr(self, f"embed_{k}")
            embed = embed_layer(code[k])  # (B, 1, C)
            embeds.append(embed)
        embeds = torch.cat(embeds, dim=1)
        embeds = embeds.transpose(0, 1)

        x = torch.cat((condition, embeds), dim=0)  # (L, B, C)
        x = self.sequence_pos_encoding(x)

        x = self.seqTransEncoder(x, mask=self.mask)

        logits = {}
        for i, k in enumerate(self.pred_keys):
            layer = getattr(self, f"pred_{k}")
            logit = layer(x[i + 1])
            logits[k] = logit
        output["logits"] = logits
        return output

    def predict_code(self, env, action, **kwargs):
        env = self.env_encoder(env)[None]
        action = self.action_encoder(action)[None]
        condition = torch.cat((env, action), dim=0)  # (2, B, C)
        preds = {}
        probs = {}
        embeds = []
        for i, k in enumerate(self.pred_keys):
            x = torch.cat((condition,) + tuple(embeds), dim=0)
            x = self.sequence_pos_encoding(x)
            x = self.seqTransEncoder(x)
            layer = getattr(self, f"pred_{k}")
            logit = layer(x[-1])
            prob = torch.softmax(logit, dim=-1)  # [B, C]
            probs[k] = prob.unsqueeze(1)
            prob = dist.categorical.Categorical(prob)
            pred = prob.sample().unsqueeze(-1)  # [B, 1]
            preds[k] = pred
            embed_layer = getattr(self, f"embed_{k}")
            embed = embed_layer(pred)  # [B, 1, C]
            embeds.append(embed.transpose(0, 1))
        return preds, probs

    def inference_forward(self, env, action, external_model, **kwargs):
        """_summary_

        Args:
            traj (tensor): [B, C]
            I (tensor): [B, C]

        Returns:
            dict(tensor):
                feats: [B, P, c]
                probs: [B, P, c]
                preds: [B, P]
        """
        env_bkp = env.clone()
        action_bkp = action.clone()
        preds, probs = self.predict_code(env, action)
        decode_output = self.external_forward2pose(
            external_model, preds, env_bkp, action_bkp, **kwargs
        )
        output = {"probs": probs, "preds": preds}
        output.update(decode_output)

        return output

    def external_forward2pose(self, external_model, pred, env, action, **kwargs):
        """_summary_

        Args:
            external_model (torch model): _description_
            pred (tensor int): [B, P]
        Returns:
            y_hat (tensor): [B, C]
        """
        data = {"code": pred, "env": env[None], "action": action[None]}
        if "I" in kwargs.keys():
            data["I"] = kwargs["I"][None]
        data = to_device(data, env.device)
        output = external_model.decode_forward(data)
        return output
