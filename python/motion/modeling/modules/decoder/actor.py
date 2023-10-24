import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from motion.modeling.operators.mlp import MLPBlock
from motion.modeling.operators.embedding import EmbeddingLayer
from motion.modeling.operators.transformer import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from motion.utils.tensor import lengths_to_mask

from motion.modeling.operators.builder import OPERATORS
from motion.modeling.modules.builder import MODULES
from motion.utils.utils import to_device


@MODULES.register_module()
class ActorAgnosticDecoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 247,
        input_pose_dim: int = 92,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        norm: str = None,
        activation: str = "ELU",
        positional_encoding_type: str = "PositionalEncoding",
        mask_token: bool = False,
        input_state_dim: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.input_pose_dim = input_pose_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm = norm
        self.dropout = dropout
        self.activation = activation
        self.positional_encoding_type = positional_encoding_type
        self.enable_mask_token = mask_token
        self.input_state_dim = input_state_dim
        self._build_model()

    def _build_model(self):
        self._build_encoder_network()
        self._build_transformer()
        self._build_final_layer()

    def _build_encoder_network(self):
        self.INet = nn.Sequential(
            MLPBlock(
                2048,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.trajNet = nn.Sequential(
            MLPBlock(
                self.state_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.poseNet = nn.Sequential(
            MLPBlock(
                self.input_pose_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.base = nn.Linear(3 * self.latent_dim, self.latent_dim)

    def _build_transformer(self):
        if self.enable_mask_token:
            self.mask_token = nn.Parameter(torch.randn(1, self.latent_dim))
        positional_encoding = OPERATORS.get(self.positional_encoding_type)
        self.sequence_pos_encoding = positional_encoding(self.latent_dim, self.dropout)

        seq_trans_decoder_layer = TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
        )

        self.seqTransDecoder = nn.TransformerDecoder(
            seq_trans_decoder_layer, num_layers=self.num_layers
        )

    def _build_final_layer(self):
        self.final_layer = nn.Linear(self.latent_dim, self.input_pose_dim)

    def _final_layer_forward(self, x):
        """
        Args:
            x (tensor): [L, B, C]

        Returns:
            dict (dict(tensor)):
                'y_hat': [B, L, C']
        """
        # Pytorch Transformer: [Sequence, Batch size, ...]
        y_hat = x.permute(1, 0, 2)
        return {"y_hat": y_hat}

    def pre_forward(self, I, traj, static_pose, lengths, mask):
        """
        Args:
            I (tensor): [B, L, C]
            traj (tensor): [B, L, C]
            static_pose (tensor): [B, L, C], only the start and the end are non-zero
            lengths (List(int)): the length of each data in the batch
            mask (bool tensor): [B, L]

        Returns:
            time_queries (tensor): [L, B, C]
            mask (bool tensor): [B, L]
        """
        I = self.INet(I)
        traj = self.trajNet(traj)
        # I = torch.zeros_like(I)
        # traj = torch.zeros_like(traj)
        static_pose = self.poseNet(static_pose)
        x = torch.cat((I, traj, static_pose), dim=-1)
        # x = torch.zeros_like(x)
        x = self.base(x)

        if lengths is None:
            lengths = [len(I_) for I_ in I]
        if mask is None:
            mask = lengths_to_mask(lengths, I.device)

        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Construct time queries
        time_queries = self.sequence_pos_encoding(x)

        return time_queries, mask

    def forward(
        self,
        I,
        traj,
        static_pose,
        z,
        lengths=None,
        mask=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            I (tensor): [B, L, C]
            traj (tensor): [B, L, C]
            static_pose (tensor): [B, L, C], only the start and the end are non-zero
            z (tensor): [B, C]
            lengths (List(int)): the length of each data in the batch
            mask (bool tensor): [B, L]

        Returns:
            dict (dict(tensor)):
                'y_hat': [B, L, C]
        """
        time_queries, mask = self.pre_forward(I, traj, static_pose, lengths, mask)

        z = z[None]  # sequence of 1 element for the memory

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(
            tgt=time_queries, memory=z, tgt_key_padding_mask=~mask
        )

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        return self._final_layer_forward(output)

    def inference_forward(self, *args, **kwargs):
        return self(*args, **kwargs)

    def merge_predictions(self, results, lengths):
        preds = []
        for i in range(len(lengths)):
            if "y_hat" in results.keys():
                pred = results["y_hat"]
            else:
                pred = results["logits"]
            preds.append(pred[i, : lengths[i]])  # [L, C]
            if i < len(lengths) - 1:
                preds[-1] = preds[-1][:-1]
        preds = torch.cat(preds, dim=0)
        return preds


@MODULES.register_module()
class ActorAgnosticEnvDecoder(ActorAgnosticDecoder):
    def __init__(
        self,
        state_dim: int = 247,
        input_pose_dim: int = 92,
        env_dim: int = 315,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        norm: str = None,
        activation: str = "ELU",
        positional_encoding_type: str = "PositionalEncoding",
        mask_token: bool = False,
        **kwargs,
    ):
        self.env_dim = env_dim
        super().__init__(
            state_dim,
            input_pose_dim,
            latent_dim,
            ff_size,
            num_layers,
            num_heads,
            dropout,
            norm,
            activation,
            positional_encoding_type,
            mask_token,
            **kwargs,
        )

    def _build_encoder_network(self):
        super()._build_encoder_network()
        self.envNet = nn.Sequential(
            MLPBlock(
                self.env_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.base = nn.Linear(4 * self.latent_dim, self.latent_dim)

    def pre_forward(self, I, env, traj, static_pose, lengths, mask):
        """
        Args:
            I (tensor): [B, L, C]
            traj (tensor): [B, L, C]
            static_pose (tensor): [B, L, C], only the start and the end are non-zero
            lengths (List(int)): the length of each data in the batch
            mask (bool tensor): [B, L]

        Returns:
            time_queries (tensor): [L, B, C]
            mask (bool tensor): [B, L]
        """
        I = self.INet(I)
        env = self.envNet(env)
        traj = self.trajNet(traj)
        # I = torch.zeros_like(I)
        # traj = torch.zeros_like(traj)
        # env = torch.zeros_like(env)
        static_pose = self.poseNet(static_pose)
        x = torch.cat((I, env, traj, static_pose), dim=-1)
        # x = torch.zeros_like(x)
        x = self.base(x)

        if lengths is None:
            lengths = [len(I_) for I_ in I]
        if mask is None:
            mask = lengths_to_mask(lengths, I.device)

        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Construct time queries
        time_queries = self.sequence_pos_encoding(x)

        return time_queries, mask

    def forward(
        self,
        I,
        env,
        traj,
        static_pose,
        z,
        lengths=None,
        mask=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            I (tensor): [B, L, C]
            traj (tensor): [B, L, C]
            static_pose (tensor): [B, L, C], only the start and the end are non-zero
            z (tensor): [B, C]
            lengths (List(int)): the length of each data in the batch
            mask (bool tensor): [B, L]

        Returns:
            dict (dict(tensor)):
                'y_hat': [B, L, C]
        """
        time_queries, mask = self.pre_forward(I, env, traj, static_pose, lengths, mask)

        z = z[None]  # sequence of 1 element for the memory
        # z = torch.zeros_like(z)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(
            tgt=time_queries, memory=z, tgt_key_padding_mask=~mask
        )

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        return self._final_layer_forward(output)


@MODULES.register_module()
class ActorTrajDecoder(ActorAgnosticDecoder):
    def __init__(
        self,
        state_dim: int = 122,
        input_pose_dim: int = 264,
        latent_dim: int = 256,
        traj_dim: int = 4,
        root_dim: int = 4,
        goal_dim: int = 6,
        pred_time_range: int = 900,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        norm: str = None,
        activation: str = "ELU",
        positional_encoding_type: str = "PositionalEncoding",
        mask_token: bool = False,
        attention_op: str = "MHA",
        **kwargs,
    ):
        self.traj_dim = traj_dim
        self.root_dim = root_dim
        self.goal_dim = goal_dim
        self.pred_time_range = pred_time_range
        self.attention_op = attention_op
        super().__init__(
            state_dim,
            input_pose_dim,
            latent_dim,
            ff_size,
            num_layers,
            num_heads,
            dropout,
            norm,
            activation,
            positional_encoding_type,
            mask_token,
            **kwargs,
        )

    def _build_encoder_network(self):
        self.ItoRNet = nn.Sequential(
            MLPBlock(
                2048,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.ItoGNet = nn.Sequential(
            MLPBlock(
                2048,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.poseNet = nn.Sequential(
            MLPBlock(
                self.input_pose_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, activation=self.activation),
        )
        self.rootNet = nn.Sequential(
            MLPBlock(
                self.root_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.goalNet = nn.Sequential(
            MLPBlock(
                self.goal_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.tNet = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim))

    def _build_tokens(self):
        positional_encoding = OPERATORS.get(self.positional_encoding_type)
        self.sequence_pos_encoding = positional_encoding(self.latent_dim, self.dropout)
        self.t_token = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        self.length_encoding = positional_encoding(self.latent_dim, 0.0)

    def _build_transformer(self):
        self._build_tokens()
        if self.attention_op == "MHA":
            attention_op = None
        elif self.attention_op == "MHA":
            attention_op = MHA
        else:
            raise NotImplementedError

        seq_trans_decoder_layer = TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            attention_op=attention_op,
        )

        self.seqTransDecoder = nn.TransformerDecoder(
            seq_trans_decoder_layer, num_layers=self.num_layers
        )

    def _build_final_layer(self):
        self.t_predlayer = nn.Linear(self.latent_dim, self.pred_time_range)
        self.traj_predlayer = nn.Linear(self.latent_dim, self.traj_dim)
        self.state_predlayer = nn.Linear(self.latent_dim, self.state_dim)

    def extract_feature(self, pose, ItoR, ItoG, goal, root, *args, **kwargs):
        pose = self.poseNet(pose)
        ItoR = self.ItoRNet(ItoR)
        ItoG = self.ItoGNet(ItoG)
        goal = self.goalNet(goal)
        root = self.rootNet(root)

        conditions = torch.cat((pose, ItoR, ItoG, root, goal), dim=1)
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(
        self,
        pose,
        ItoR,
        ItoG,
        goal,
        root,
        y,
        z,
        lengths=None,
        mask=None,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            pose (_type_): [B, 1, C]
            ItoR (_type_): [B, 1, C]
            ItoG (_type_): [B, 1, C]
            goal (_type_): [B, 1, C]
            root (_type_): [B, 1, C]
            y (_type_): [B, T, C]
            z (tensor): [B, C]
            lengths (List(int)): the length of each data in the batch
            mask (bool tensor): [B, L], Defaults to None.

        Returns:
            dict (dict(tensor)):
                'y_hat': [B, L, C]
        """
        bs = pose.shape[0]
        conditions = self.extract_feature(pose, ItoR, ItoG, goal, root)
        t_token = self.t_token.expand(-1, bs, -1)  # 1, b, c

        tseq = torch.cat((t_token, conditions), 0)
        tseq = tseq + self.sequence_pos_encoding(tseq)

        z = z[None]  # sequence of 1 element for the memory

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=tseq, memory=z)[0]

        pred_t = self.t_predlayer(output)

        if self.training:
            prob = torch.softmax(pred_t, dim=-1)  # [B, c]
            replace_mask = torch.rand(bs, device=prob.device) < self.dropout
            prob = dist.categorical.Categorical(prob)
            lengths_sample = prob.sample()  # [B]
            lengths[replace_mask] = lengths_sample[replace_mask]
        t_embed = self.length_encoding.query(lengths)
        t_embed = self.tNet(t_embed).permute(1, 0, 2)  # 1, b, c

        traj = torch.zeros((y.shape[1], bs, self.latent_dim), device=pose.device)
        xseq = torch.cat((t_embed, conditions, traj), dim=0)
        xseq = xseq + self.sequence_pos_encoding(xseq)

        extra_num = conditions.shape[0]
        token_mask = torch.ones((bs, 1 + extra_num), dtype=bool, device=pose.device)
        aug_mask = torch.cat((token_mask, mask), 1)
        output = self.seqTransDecoder(
            tgt=xseq, memory=z, tgt_key_padding_mask=~aug_mask
        )
        # zero for padded area
        output[~aug_mask.T] = 0
        output = output.permute(1, 0, 2)

        pred_traj = self.traj_predlayer(output)[:, 1 + extra_num :]
        pred_y = self.state_predlayer(output)[:, 1 + extra_num :]
        return {"y_hat": pred_y, "t": pred_t, "traj": pred_traj}

    def inference_forward(self, data, *args, **kwargs):
        z = data["z"]
        bs = z.shape[0]
        conditions = self.extract_feature(**data)

        t_token = self.t_token.expand(-1, bs, -1)  # 1, b, c

        tseq = torch.cat((t_token, conditions), 0)
        tseq = tseq + self.sequence_pos_encoding(tseq)

        z = z[None]  # sequence of 1 element for the memory
        # z = z * 0.1

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=tseq, memory=z)[0]

        pred_t = self.t_predlayer(output)
        # We always predict more than 2 Milestones, including start and end.
        # So we ommit the 0 and 1
        pred_t[:, 0] = -1e9
        pred_t[:, 1] = -1e9
        # prior_t = data["prior_t"]
        # for i in range(prior_t):
        #     pred_t[:, i] = -1e9
        prob = torch.softmax(pred_t, dim=-1)  # [B, c]
        # lengths = prob.argmax(dim=-1)  # [B]
        prob = dist.categorical.Categorical(prob)
        lengths = prob.sample()  # [B]
        # lengths[0] = 372

        traj = torch.zeros((lengths, bs, self.latent_dim), device=z.device)
        t_embed = self.length_encoding.query(lengths)
        t_embed = self.tNet(t_embed).permute(1, 0, 2)  # 1, b, c

        xseq = torch.cat((t_embed, conditions, traj), dim=0)
        xseq = xseq + self.sequence_pos_encoding(xseq)

        output = self.seqTransDecoder(tgt=xseq, memory=z)
        output = output.permute(1, 0, 2)
        extra_num = conditions.shape[0] + 1
        pred_traj = self.traj_predlayer(output)[:, extra_num:]
        pred_y = self.state_predlayer(output)[:, extra_num:]

        return {"traj": pred_traj, "state": pred_y, "prob": prob.probs}


@MODULES.register_module()
class ActorTrajEnvDecoder(ActorTrajDecoder):
    def __init__(
        self,
        state_dim: int = 122,
        input_pose_dim: int = 264,
        env_dim: int = 315,
        latent_dim: int = 256,
        traj_dim: int = 4,
        root_dim: int = 4,
        goal_dim: int = 6,
        pred_time_range: int = 900,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        norm: str = None,
        activation: str = "ELU",
        positional_encoding_type: str = "PositionalEncoding",
        mask_token: bool = False,
        attention_op: str = "MHA",
        **kwargs,
    ):
        self.env_dim = env_dim
        super().__init__(
            state_dim,
            input_pose_dim,
            latent_dim,
            traj_dim,
            root_dim,
            goal_dim,
            pred_time_range,
            ff_size,
            num_layers,
            num_heads,
            dropout,
            norm,
            activation,
            positional_encoding_type,
            mask_token,
            attention_op,
            **kwargs,
        )

    def _build_encoder_network(self):
        super()._build_encoder_network()
        self.envNet = nn.Sequential(
            MLPBlock(
                self.env_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )

    def extract_feature(self, pose, ItoR, ItoG, env, goal, root, *args, **kwargs):
        pose = self.poseNet(pose)
        ItoR = self.ItoRNet(ItoR)
        ItoG = self.ItoGNet(ItoG)
        env = self.envNet(env)
        goal = self.goalNet(goal)
        root = self.rootNet(root)

        conditions = torch.cat((pose, ItoR, ItoG, env, root, goal), dim=1)
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(
        self,
        pose,
        ItoR,
        ItoG,
        env,
        goal,
        root,
        y,
        z,
        lengths=None,
        mask=None,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            pose (_type_): [B, 1, C]
            ItoR (_type_): [B, 1, C]
            ItoG (_type_): [B, 1, C]
            goal (_type_): [B, 1, C]
            root (_type_): [B, 1, C]
            y (_type_): [B, T, C]
            z (tensor): [B, C]
            lengths (List(int)): the length of each data in the batch
            mask (bool tensor): [B, L], Defaults to None.

        Returns:
            dict (dict(tensor)):
                'y_hat': [B, L, C]
        """
        bs = pose.shape[0]
        conditions = self.extract_feature(pose, ItoR, ItoG, env, goal, root)
        t_token = self.t_token.expand(-1, bs, -1)  # 1, b, c

        tseq = torch.cat((t_token, conditions), 0)
        tseq = tseq + self.sequence_pos_encoding(tseq)

        z = z[None]  # sequence of 1 element for the memory

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=tseq, memory=z)[0]

        pred_t = self.t_predlayer(output)

        if self.training:
            prob = torch.softmax(pred_t, dim=-1)  # [B, c]
            replace_mask = torch.rand(bs, device=prob.device) < self.dropout
            prob = dist.categorical.Categorical(prob)
            lengths_sample = prob.sample()  # [B]
            lengths[replace_mask] = lengths_sample[replace_mask]
        t_embed = self.length_encoding.query(lengths)
        t_embed = self.tNet(t_embed).permute(1, 0, 2)  # 1, b, c

        traj = torch.zeros((y.shape[1], bs, self.latent_dim), device=pose.device)
        xseq = torch.cat((t_embed, conditions, traj), dim=0)
        xseq = xseq + self.sequence_pos_encoding(xseq)

        extra_num = conditions.shape[0]
        token_mask = torch.ones((bs, 1 + extra_num), dtype=bool, device=pose.device)
        aug_mask = torch.cat((token_mask, mask), 1)
        output = self.seqTransDecoder(
            tgt=xseq, memory=z, tgt_key_padding_mask=~aug_mask
        )
        # zero for padded area
        output[~aug_mask.T] = 0
        output = output.permute(1, 0, 2)

        pred_traj = self.traj_predlayer(output)[:, 1 + extra_num :]
        pred_y = self.state_predlayer(output)[:, 1 + extra_num :]
        return {"y_hat": pred_y, "t": pred_t, "traj": pred_traj}


@MODULES.register_module()
class ActorTrajCompletionDecoder(ActorTrajDecoder):
    def _build_tokens(self):
        positional_encoding = OPERATORS.get(self.positional_encoding_type)
        self.sequence_pos_encoding = positional_encoding(self.latent_dim, self.dropout)

    def _build_encoder_network(self):
        self.ItoRNet = nn.Sequential(
            MLPBlock(
                2048,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.ItoGNet = nn.Sequential(
            MLPBlock(
                2048,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.bistateNet = nn.Sequential(
            MLPBlock(
                self.input_state_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.rootNet = nn.Sequential(
            MLPBlock(
                self.root_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.goalNet = nn.Sequential(
            MLPBlock(
                self.goal_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )

    def _build_final_layer(self):
        self.traj_predlayer = nn.Linear(self.latent_dim, self.traj_dim)
        self.state_predlayer = nn.Linear(self.latent_dim, self.state_dim)

    def extract_feature(
        self, start_state, end_state, ItoR, ItoG, goal, root, *args, **kwargs
    ):
        start_state = self.bistateNet(start_state)
        end_state = self.bistateNet(end_state)
        ItoR = self.ItoRNet(ItoR)
        ItoG = self.ItoGNet(ItoG)
        goal = self.goalNet(goal)
        root = self.rootNet(root)

        conditions = torch.cat((start_state, end_state, ItoR, ItoG, root, goal), dim=1)
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(
        self,
        start_state,
        end_state,
        ItoR,
        ItoG,
        goal,
        root,
        z,
        y=None,
        L=None,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            start_state (_type_): [B, 1, C]
            end_state (_type_): [B, 1, C]
            ItoR (_type_): [B, 1, C]
            ItoG (_type_): [B, 1, C]
            goal (_type_): [B, 1, C]
            root (_type_): [B, 1, C]
            y (_type_): [B, T, C]
            z (tensor): [B, C]
            L (int):

        Returns:
            dict (dict(tensor)):
                'y_hat': [B, L, C]
        """
        bs = root.shape[0]
        conditions = self.extract_feature(
            start_state, end_state, ItoR, ItoG, goal, root
        )
        z = z[None]  # sequence of 1 element for the memory

        # Pass through the transformer decoder
        # with the latent vector for memory
        if y is not None:
            L = y.shape[1]
        traj = torch.zeros((L, bs, self.latent_dim), device=root.device)
        xseq = torch.cat((conditions, traj), dim=0)
        xseq = xseq + self.sequence_pos_encoding(xseq)

        output = self.seqTransDecoder(tgt=xseq, memory=z)
        # zero for padded area
        output = output.permute(1, 0, 2)
        pred_traj = self.traj_predlayer(output)[:, -L:]
        pred_y = self.state_predlayer(output)[:, -L:]
        return {"y_hat": pred_y, "traj": pred_traj}

    def inference_forward(self, *args, **kwargs):
        return self(*args, **kwargs)


@MODULES.register_module()
class ActorTrajCompletionEnvDecoder(ActorTrajCompletionDecoder):
    def __init__(
        self,
        state_dim: int = 122,
        input_pose_dim: int = 264,
        env_dim: int = 315,
        latent_dim: int = 256,
        traj_dim: int = 4,
        root_dim: int = 4,
        goal_dim: int = 6,
        pred_time_range: int = 900,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        norm: str = None,
        activation: str = "ELU",
        positional_encoding_type: str = "PositionalEncoding",
        mask_token: bool = False,
        attention_op: str = "MHA",
        **kwargs,
    ):
        self.env_dim = env_dim
        super().__init__(
            state_dim,
            input_pose_dim,
            latent_dim,
            traj_dim,
            root_dim,
            goal_dim,
            pred_time_range,
            ff_size,
            num_layers,
            num_heads,
            dropout,
            norm,
            activation,
            positional_encoding_type,
            mask_token,
            attention_op,
            **kwargs,
        )

    def _build_encoder_network(self):
        super()._build_encoder_network()
        self.envNet = nn.Sequential(
            MLPBlock(
                self.env_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )

    def extract_feature(
        self, start_state, end_state, ItoR, ItoG, env, goal, root, *args, **kwargs
    ):
        start_state = self.bistateNet(start_state)
        end_state = self.bistateNet(end_state)
        ItoR = self.ItoRNet(ItoR)
        ItoG = self.ItoGNet(ItoG)
        env = self.envNet(env)
        goal = self.goalNet(goal)
        root = self.rootNet(root)

        # conditions = torch.cat((start_state, end_state, ItoR, ItoG, root, goal), dim=1)
        conditions = torch.cat(
            (start_state, end_state, ItoR, ItoG, root, goal, env), dim=1
        )
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(
        self,
        start_state,
        end_state,
        ItoR,
        ItoG,
        env,
        goal,
        root,
        z,
        y=None,
        L=None,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            start_state (_type_): [B, 1, C]
            end_state (_type_): [B, 1, C]
            ItoR (_type_): [B, 1, C]
            ItoG (_type_): [B, 1, C]
            goal (_type_): [B, 1, C]
            root (_type_): [B, 1, C]
            y (_type_): [B, T, C]
            z (tensor): [B, C]
            L (int):

        Returns:
            dict (dict(tensor)):
                'y_hat': [B, L, C]
        """
        bs = root.shape[0]
        conditions = self.extract_feature(
            start_state, end_state, ItoR, ItoG, env, goal, root
        )
        z = z[None]  # sequence of 1 element for the memory

        # Pass through the transformer decoder
        # with the latent vector for memory
        if y is not None:
            L = y.shape[1]
        traj = torch.zeros((L, bs, self.latent_dim), device=root.device)
        xseq = torch.cat((conditions, traj), dim=0)
        xseq = xseq + self.sequence_pos_encoding(xseq)

        output = self.seqTransDecoder(tgt=xseq, memory=z)
        # zero for padded area
        output = output.permute(1, 0, 2)
        pred_traj = self.traj_predlayer(output)[:, -L:]
        pred_y = self.state_predlayer(output)[:, -L:]
        return {"y_hat": pred_y, "traj": pred_traj}


@MODULES.register_module()
class ActorTrajRefineEnvDecoder(ActorTrajCompletionEnvDecoder):
    def _build_encoder_network(self):
        self.INet = nn.Sequential(
            MLPBlock(2048, 256, dropout=self.dropout, activation=self.activation),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
            MLPBlock(256, self.latent_dim, activation=self.activation),
        )
        self.trajNet = nn.Sequential(
            MLPBlock(
                self.traj_dim, 256, dropout=self.dropout, activation=self.activation
            ),
            MLPBlock(256, self.latent_dim, activation=self.activation),
        )
        self.rootgoalNet = nn.Sequential(
            MLPBlock(
                self.root_dim, 256, dropout=self.dropout, activation=self.activation
            ),
            MLPBlock(256, self.latent_dim, activation=self.activation),
        )
        self.envNet = nn.Sequential(
            MLPBlock(
                self.env_dim, 256, dropout=self.dropout, activation=self.activation
            ),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
            MLPBlock(256, self.latent_dim, activation=self.activation),
        )

    def extract_feature(self, I, env, goal, root, *args, **kwargs):
        I = self.INet(I)
        env = self.envNet(env)
        goal = self.rootgoalNet(goal)
        root = self.rootgoalNet(root)

        conditions = torch.cat((I, root, goal, env), dim=1)
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(
        self,
        I,
        env,
        goal,
        root,
        init_traj,
        z,
        y=None,
        L=None,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            start_state (_type_): [B, 1, C]
            end_state (_type_): [B, 1, C]
            ItoR (_type_): [B, 1, C]
            ItoG (_type_): [B, 1, C]
            goal (_type_): [B, 1, C]
            root (_type_): [B, 1, C]
            y (_type_): [B, T, C]
            z (tensor): [B, C]
            L (int):

        Returns:
            dict (dict(tensor)):
                'y_hat': [B, L, C]
        """
        bs = root.shape[0]
        conditions = self.extract_feature(I, env, goal, root)
        z = z[None]  # sequence of 1 element for the memory

        # Pass through the transformer decoder
        # with the latent vector for memory
        if y is not None:
            L = y.shape[1]
        init_traj = self.trajNet(init_traj).permute(1, 0, 2)
        xseq = torch.cat((conditions, init_traj), dim=0)
        xseq = self.sequence_pos_encoding(xseq)

        output = self.seqTransDecoder(tgt=xseq, memory=z)
        # zero for padded area
        output = output.permute(1, 0, 2)
        pred_traj = self.traj_predlayer(output)[:, -init_traj.shape[0] :]
        return {"y_hat": pred_traj}


@MODULES.register_module()
class ActorMilestonePoseDecoder(ActorTrajCompletionDecoder):
    def _build_tokens(self):
        positional_encoding = OPERATORS.get(self.positional_encoding_type)
        self.sequence_pos_encoding = positional_encoding(self.latent_dim, self.dropout)

    def _build_encoder_network(self):
        self.INet = nn.Sequential(
            MLPBlock(
                2048,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.trajNet = nn.Sequential(
            MLPBlock(
                self.traj_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.poseNet = nn.Sequential(
            MLPBlock(
                self.state_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )

    def _build_final_layer(self):
        self.pose_predlayer = nn.Linear(self.latent_dim, self.state_dim)

    def extract_feature(self, pose, I, traj, *args, **kwargs):
        pose = self.poseNet(pose)  # [B, 1, C]
        I = self.INet(I)  # [B, L, C]
        traj = self.trajNet(traj)  # [B, L, C]
        x = I + traj
        conditions = torch.cat((pose, x), dim=1)
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(self, pose, I, traj, z, lengths=None, mask=None, *args, **kwargs):
        """_summary_

        Args:
            pose (_type_): [B, 1, C]
            xx : [B, L, C]
            y (_type_): [B, L, C]
            z (tensor): [B, C]
            L (int):

        Returns:
            dict (dict(tensor)):
                'y_hat': [B, L, C]
        """
        conditions = self.extract_feature(pose, I, traj)
        z = z[None]  # sequence of 1 element for the memory

        if lengths is None:
            lengths = [len(I_) for I_ in I]
        if mask is None:
            mask = lengths_to_mask(lengths, I.device)
        token_mask = torch.ones(pose.shape[:2], dtype=bool, device=z.device)
        aug_mask = torch.cat((token_mask, mask), 1)
        xseq = conditions
        # Pass through the transformer decoder
        # with the latent vector for memory
        xseq = xseq + self.sequence_pos_encoding(xseq)

        output = self.seqTransDecoder(
            tgt=xseq, memory=z, tgt_key_padding_mask=~aug_mask
        )
        # zero for padded area
        output = output.permute(1, 0, 2)
        traj_num = traj.shape[1]
        pred_pose = self.pose_predlayer(output)[:, -traj_num:]
        return {"y_hat": pred_pose}

    def inference_forward(self, data, **kwargs):
        return self(**data, **kwargs)


@MODULES.register_module()
class ActorMilestonePoseEnvDecoder(ActorMilestonePoseDecoder):
    def __init__(
        self,
        state_dim: int = 122,
        input_pose_dim: int = 264,
        env_dim: int = 315,
        latent_dim: int = 256,
        traj_dim: int = 4,
        root_dim: int = 4,
        goal_dim: int = 6,
        pred_time_range: int = 900,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        norm: str = None,
        activation: str = "ELU",
        positional_encoding_type: str = "PositionalEncoding",
        mask_token: bool = False,
        attention_op: str = "MHA",
        **kwargs,
    ):
        self.env_dim = env_dim
        super().__init__(
            state_dim,
            input_pose_dim,
            latent_dim,
            traj_dim,
            root_dim,
            goal_dim,
            pred_time_range,
            ff_size,
            num_layers,
            num_heads,
            dropout,
            norm,
            activation,
            positional_encoding_type,
            mask_token,
            attention_op,
            **kwargs,
        )

    def _build_encoder_network(self):
        super()._build_encoder_network()
        self.envNet = nn.Sequential(
            MLPBlock(
                self.env_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(
                256,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )

    def extract_feature(self, pose, I, env, traj, *args, **kwargs):
        pose = self.poseNet(pose)  # [B, 1, C]
        I = self.INet(I)  # [B, L, C]
        env = self.envNet(env)
        traj = self.trajNet(traj)  # [B, L, C]
        x = I + traj + env
        conditions = torch.cat((pose, x), dim=1)
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(self, pose, I, env, traj, z, lengths=None, mask=None, *args, **kwargs):
        """_summary_

        Args:
            pose (_type_): [B, 1, C]
            xx : [B, L, C]
            y (_type_): [B, L, C]
            z (tensor): [B, C]
            L (int):

        Returns:
            dict (dict(tensor)):
                'y_hat': [B, L, C]
        """
        conditions = self.extract_feature(pose, I, env, traj)
        z = z[None]  # sequence of 1 element for the memory

        if lengths is None:
            lengths = [len(I_) for I_ in I]
        if mask is None:
            mask = lengths_to_mask(lengths, I.device)
        token_mask = torch.ones(pose.shape[:2], dtype=bool, device=z.device)
        aug_mask = torch.cat((token_mask, mask), 1)
        xseq = conditions
        # Pass through the transformer decoder
        # with the latent vector for memory
        xseq = xseq + self.sequence_pos_encoding(xseq)

        output = self.seqTransDecoder(
            tgt=xseq, memory=z, tgt_key_padding_mask=~aug_mask
        )
        # zero for padded area
        output = output.permute(1, 0, 2)
        traj_num = traj.shape[1]
        pred_pose = self.pose_predlayer(output)[:, -traj_num:]
        return {"y_hat": pred_pose}
