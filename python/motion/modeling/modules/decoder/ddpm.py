import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from motion.modeling.operators.mlp import MLPBlock, ResTimeMLPBlock
from motion.modeling.operators.embedding import EmbeddingLayer
from motion.modeling.operators.positional_encoding import PositionalEncoding
from motion.utils.tensor import lengths_to_mask
from motion.modeling.modules.decoder.actor import (
    ActorMilestonePoseEnvDecoder,
    ActorTrajCompletionEnvDecoder,
    ActorTrajEnvDecoder,
    ActorAgnosticEnvDecoder,
)

from motion.modeling.operators.builder import OPERATORS
from motion.modeling.modules.builder import MODULES
from motion.utils.utils import to_device


@MODULES.register_module()
class DDPMEnvDecoder(ActorAgnosticEnvDecoder):
    def __init__(
        self,
        self_condition: bool = True,
        cond_mask_prob: float = 0.1,
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
        self.self_condition = self_condition
        self.cond_mask_prob = cond_mask_prob
        super().__init__(
            state_dim,
            input_pose_dim,
            env_dim,
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
        input_channels = self.input_pose_dim
        if self.self_condition:
            input_channels = input_channels * 2
        self.init_layer = nn.Linear(input_channels, self.latent_dim)
        self.time_mlp = nn.Sequential(
            MLPBlock(
                self.latent_dim,
                self.latent_dim,
                norm=self.norm,
                activation=self.activation,
            ),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def _build_final_layer(self):
        super()._build_final_layer()
        self.res_layer = ResTimeMLPBlock(
            self.latent_dim * 2,
            self.latent_dim,
            self.latent_dim,
            norm=self.norm,
        )

    def pre_forward(self, I, env, traj, static_pose, lengths, mask, uncond=False):
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
        static_pose = self.poseNet(static_pose)
        x = torch.cat((I, env, traj, static_pose), dim=-1)
        x = self.base(x)

        if lengths is None:
            lengths = [len(I_) for I_ in I]
        if mask is None:
            mask = lengths_to_mask(lengths, I.device)

        if uncond:
            x = torch.zeros_like(x)
        elif self.training and self.cond_mask_prob > 0.0:
            bs = x.shape[0]
            mask = torch.bernoulli(
                torch.ones(bs, device=x.device) * self.cond_mask_prob
            ).view(bs, 1, 1)
            x = x * (1.0 - mask)
        else:
            pass

        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]
        # Construct time queries
        time_queries = self.sequence_pos_encoding(x)

        return time_queries, mask

    def forward(
        self,
        x,
        t,
        x_self_cond,
        I,
        env,
        traj,
        static_pose,
        lengths=None,
        mask=None,
        uncond=False,
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
        time_queries, mask = self.pre_forward(
            I, env, traj, static_pose, lengths, mask, uncond
        )

        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim=-1)
        x = self.init_layer(x)  # b, l, c
        res = x.clone()
        x = x.permute(1, 0, 2)  # l, b, c
        time_embed = self.sequence_pos_encoding.query(t)
        time_embed = self.time_mlp(time_embed)  # b, 1, c
        time_embed = time_embed.permute(1, 0, 2)  # 1, b, c
        time_queries = time_queries + x

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=time_queries, memory=time_embed)

        output = output.permute(1, 0, 2)  # b, l, c
        output = torch.cat((res, output), dim=-1)
        output = self.res_layer(output, time_embed.permute(1, 0, 2))
        output = self.final_layer(output)
        return {"pred": output, "y_hat": output}

    def inference_forward(self, *args, **kwargs):
        return self(*args, **kwargs)

    def get_shape(self, data):
        shape = data["static_pose"].shape
        return shape


@MODULES.register_module()
class DDPMTrajEnvDecoder(ActorTrajEnvDecoder):
    def __init__(
        self,
        self_condition: bool = True,
        cond_mask_prob: float = 0.1,
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
        self.self_condition = self_condition
        self.cond_mask_prob = cond_mask_prob
        super().__init__(
            state_dim,
            input_pose_dim,
            env_dim,
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
        input_channels = self.traj_dim + self.state_dim
        if self.self_condition:
            input_channels = input_channels * 2
        self.init_layer = nn.Linear(input_channels, self.latent_dim)
        self.time_mlp = nn.Sequential(
            MLPBlock(
                self.latent_dim,
                self.latent_dim,
                norm=self.norm,
                activation=self.activation,
            ),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def _build_final_layer(self):
        self.t_predlayer = nn.Linear(self.latent_dim, self.pred_time_range)
        self.res_layer = ResTimeMLPBlock(
            self.latent_dim * 2,
            self.latent_dim,
            self.latent_dim,
            norm=self.norm,
        )
        self.predlayer = nn.Linear(self.latent_dim, self.traj_dim + self.state_dim)

    def extract_feature(
        self,
        pose,
        ItoR,
        ItoG,
        env,
        goal,
        root,
        uncond=False,
        length_pred=False,
        **kwargs,
    ):
        pose = self.poseNet(pose)
        ItoR = self.ItoRNet(ItoR)
        ItoG = self.ItoGNet(ItoG)
        env = self.envNet(env)
        goal = self.goalNet(goal)
        root = self.rootNet(root)

        conditions = torch.cat((pose, ItoR, ItoG, env, root, goal), dim=1)
        if length_pred:
            pass
        elif uncond:
            conditions = torch.zeros_like(conditions)
        elif self.training and self.cond_mask_prob > 0.0:
            bs = conditions.shape[0]
            mask = torch.bernoulli(
                torch.ones(bs, device=conditions.device) * self.cond_mask_prob
            ).view(bs, 1, 1)
            conditions = conditions * (1.0 - mask)
        else:
            pass
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(
        self,
        x,
        t,
        x_self_cond,
        pose,
        ItoR,
        ItoG,
        env,
        goal,
        root,
        lengths=None,
        mask=None,
        uncond=False,
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
        conditions = self.extract_feature(
            pose, ItoR, ItoG, env, goal, root, length_pred=True
        )
        t_token = self.t_token.expand(-1, bs, -1)  # 1, b, c

        tseq = torch.cat((t_token, conditions), 0)
        tseq = self.sequence_pos_encoding(tseq)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=tseq, memory=torch.zeros_like(t_token))[0]

        pred_t = self.t_predlayer(output)

        if self.training:
            prob = torch.softmax(pred_t, dim=-1)  # [B, c]
            replace_mask = torch.rand(bs, device=prob.device) < self.dropout
            prob = dist.categorical.Categorical(prob)
            lengths_sample = prob.sample()  # [B]
            lengths[replace_mask] = lengths_sample[replace_mask]

        t_embed = self.length_encoding.query(lengths)
        t_embed = self.tNet(t_embed).permute(1, 0, 2)  # 1, b, c

        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim=-1)
        x = self.init_layer(x)  # b, l, c
        res = x.clone()
        x = x.permute(1, 0, 2)  # l, b, c
        time_embed = self.sequence_pos_encoding.query(t)
        time_embed = self.time_mlp(time_embed)  # b, 1, c
        time_embed = time_embed.permute(1, 0, 2)  # 1, b, c

        conditions = self.extract_feature(
            pose, ItoR, ItoG, env, goal, root, uncond=uncond
        )

        xseq = torch.cat((t_embed, conditions, x), dim=0)
        xseq = self.sequence_pos_encoding(xseq)

        extra_num = conditions.shape[0]
        token_mask = torch.ones((bs, 1 + extra_num), dtype=bool, device=pose.device)
        aug_mask = torch.cat((token_mask, mask), 1)
        output = self.seqTransDecoder(
            tgt=xseq, memory=time_embed, tgt_key_padding_mask=~aug_mask
        )
        # zero for padded area
        output[~aug_mask.T] = 0
        output = output.permute(1, 0, 2)  # b, l, c
        output = output[:, 1 + extra_num :]
        output = torch.cat((res, output), dim=-1)
        output = self.res_layer(output, time_embed.permute(1, 0, 2))
        pred = self.predlayer(output)

        return {"pred": pred, "t": pred_t}

    def inference_forward(
        self,
        x,
        t,
        x_self_cond,
        pose,
        ItoR,
        ItoG,
        env,
        goal,
        root,
        lengths=None,
        mask=None,
        uncond=False,
        **kwargs,
    ):
        bs = pose.shape[0]
        t_embed = self.length_encoding.query(lengths)
        t_embed = self.tNet(t_embed).permute(1, 0, 2)  # 1, b, c

        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim=-1)
        x = self.init_layer(x)  # b, l, c
        res = x.clone()
        x = x.permute(1, 0, 2)  # b, l, c
        time_embed = self.sequence_pos_encoding.query(t)
        time_embed = self.time_mlp(time_embed)  # b, 1, c
        time_embed = time_embed.permute(1, 0, 2)  # 1, b, c

        conditions = self.extract_feature(
            pose, ItoR, ItoG, env, goal, root, uncond=uncond
        )
        xseq = torch.cat((t_embed, conditions, x), dim=0)
        xseq = self.sequence_pos_encoding(xseq)

        extra_num = conditions.shape[0]
        if mask is not None:
            token_mask = torch.ones((bs, 1 + extra_num), dtype=bool, device=pose.device)
            aug_mask = torch.cat((token_mask, mask), 1)
            aug_mask = ~aug_mask
        else:
            aug_mask = None
        output = self.seqTransDecoder(
            tgt=xseq, memory=time_embed, tgt_key_padding_mask=aug_mask
        )
        # zero for padded area
        if mask is not None:
            output[aug_mask.T] = 0
        output = output.permute(1, 0, 2)  # b, l, c
        output = output[:, 1 + extra_num :]
        output = torch.cat((res, output), dim=-1)
        output = self.res_layer(output, time_embed.permute(1, 0, 2))
        pred = self.predlayer(output)
        return {"pred": pred}

    def pred_length(self, data, *args, **kwargs):
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
        bs = data["pose"].shape[0]
        conditions = self.extract_feature(**data, length_pred=True)
        t_token = self.t_token.expand(-1, bs, -1)  # 1, b, c

        tseq = torch.cat((t_token, conditions), 0)
        tseq = self.sequence_pos_encoding(tseq)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=tseq, memory=torch.zeros_like(t_token))[0]

        pred_t = self.t_predlayer(output)
        # We always predict more than 2 Milestones, including start and end.
        # So we ommit the 0 and 1
        if "prior_t" in data.keys():
            pred_t[:, 0] = -1e9
            pred_t[:, 1] = -1e9
            pred_t[:, 2] = -1e9
            prior_t = data["prior_t"]
            for i in range(prior_t):
                pass
                # pred_t[:, i] = -1e9
        prob = torch.softmax(pred_t, dim=-1)  # [B, c]
        prob = dist.categorical.Categorical(prob)
        lengths = prob.sample()  # [B]
        return {"pred_lengths": lengths, "pred_t": pred_t, "prob": prob.probs}

    def get_shape(self, data):
        if "y" in data.keys():
            # Training
            B, L, _ = data["y"].shape
        else:
            # Socket
            # data["pred_lengths"]: (tensor) [B]
            B = data["pred_lengths"].shape[0]
            L = data["pred_lengths"].max().item()
        shape = (B, L, self.traj_dim + self.state_dim)
        return shape

    def split_pred(self, pred, prefix="pred"):
        pred_traj = pred[..., : self.traj_dim]
        y_hat = pred[..., -self.state_dim :]
        if prefix != "":
            prefix += "_"
        return {prefix + "y_hat": y_hat, prefix + "traj": pred_traj}


@MODULES.register_module()
class DDPMTrajWithPoseDecoder(DDPMTrajEnvDecoder):
    def _build_encoder_network(self):
        super()._build_encoder_network()
        input_channels = self.traj_dim + self.state_dim + self.input_pose_dim
        if self.self_condition:
            input_channels = input_channels * 2
        self.init_layer = nn.Linear(input_channels, self.latent_dim)
        self.time_mlp = nn.Sequential(
            MLPBlock(
                self.latent_dim,
                self.latent_dim,
                norm=self.norm,
                activation=self.activation,
            ),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def _build_final_layer(self):
        self.t_predlayer = nn.Linear(self.latent_dim, self.pred_time_range)
        self.res_layer = ResTimeMLPBlock(
            self.latent_dim * 2,
            self.latent_dim,
            self.latent_dim,
            norm=self.norm,
        )
        self.predlayer = nn.Linear(
            self.latent_dim, self.traj_dim + self.state_dim + self.input_pose_dim
        )

    def get_shape(self, data):
        if "y" in data.keys():
            # Training
            B, L, _ = data["y"].shape
        else:
            # Socket
            # data["pred_lengths"]: (tensor) [B]
            B = data["pred_lengths"].shape[0]
            L = data["pred_lengths"].max().item()
        shape = (B, L, self.traj_dim + self.state_dim + self.input_pose_dim)
        return shape

    def split_pred(self, pred, prefix="pred"):
        pred_traj = pred[..., : self.traj_dim]
        y_hat = pred[..., self.traj_dim :]
        if prefix != "":
            prefix += "_"
        return {prefix + "y_hat": y_hat, prefix + "traj": pred_traj}


@MODULES.register_module()
class DDPMTrajCompletionEnvDecoder(ActorTrajCompletionEnvDecoder):
    def __init__(
        self,
        self_condition: bool = True,
        cond_mask_prob: float = 0.1,
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
        self.self_condition = self_condition
        self.cond_mask_prob = cond_mask_prob
        super().__init__(
            state_dim,
            input_pose_dim,
            env_dim,
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
        input_channels = self.traj_dim + self.state_dim
        if self.self_condition:
            input_channels = input_channels * 2
        self.init_layer = nn.Linear(input_channels, self.latent_dim)
        self.time_mlp = nn.Sequential(
            MLPBlock(
                self.latent_dim,
                self.latent_dim,
                norm=self.norm,
                activation=self.activation,
            ),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def _build_final_layer(self):
        self.res_layer = ResTimeMLPBlock(
            self.latent_dim * 2,
            self.latent_dim,
            self.latent_dim,
            norm=self.norm,
        )
        self.predlayer = nn.Linear(self.latent_dim, self.traj_dim + self.state_dim)

    def extract_feature(
        self,
        start_state,
        end_state,
        ItoR,
        ItoG,
        env,
        goal,
        root,
        uncond=False,
        *args,
        **kwargs,
    ):
        start_state = self.bistateNet(start_state)
        end_state = self.bistateNet(end_state)
        ItoR = self.ItoRNet(ItoR)
        ItoG = self.ItoGNet(ItoG)
        env = self.envNet(env)
        goal = self.goalNet(goal)
        root = self.rootNet(root)

        conditions = torch.cat(
            (start_state, end_state, ItoR, ItoG, root, goal, env), dim=1
        )
        if uncond:
            conditions = torch.zeros_like(conditions)
        elif self.training and self.cond_mask_prob > 0.0:
            bs = conditions.shape[0]
            mask = torch.bernoulli(
                torch.ones(bs, device=conditions.device) * self.cond_mask_prob
            ).view(bs, 1, 1)
            conditions = conditions * (1.0 - mask)
        else:
            pass
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(
        self,
        x,
        t,
        x_self_cond,
        start_state,
        end_state,
        ItoR,
        ItoG,
        env,
        goal,
        root,
        uncond=False,
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
        conditions = self.extract_feature(
            start_state, end_state, ItoR, ItoG, env, goal, root, uncond=uncond
        )

        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim=-1)
        x = self.init_layer(x)  # b, l, c
        res = x.clone()
        x = x.permute(1, 0, 2)  # l, b, c

        time_embed = self.sequence_pos_encoding.query(t)
        time_embed = self.time_mlp(time_embed)  # b, 1, c
        time_embed = time_embed.permute(1, 0, 2)  # 1, b, c

        xseq = torch.cat((conditions, x), dim=0)
        xseq = self.sequence_pos_encoding(xseq)

        output = self.seqTransDecoder(tgt=xseq, memory=time_embed)
        output = output.permute(1, 0, 2)  # b, l, c
        extra_num = conditions.shape[0]
        output = output[:, extra_num:]
        output = torch.cat((res, output), dim=-1)
        output = self.res_layer(output, time_embed.permute(1, 0, 2))
        pred = self.predlayer(output)

        return {"pred": pred}

    def inference_forward(self, *args, **kwargs):
        return self(*args, **kwargs)

    def get_shape(self, data):
        if "y" in data.keys():
            # Training
            B, L, _ = data["y"].shape
        else:
            # Socket
            B = data["env"].shape[0]
            L = data["L"]
        shape = (B, L, self.traj_dim + self.state_dim)
        return shape

    def split_pred(self, pred, prefix="pred"):
        pred_traj = pred[..., : self.traj_dim]
        y_hat = pred[..., -self.state_dim :]
        if prefix != "":
            prefix += "_"
        return {prefix + "y_hat": y_hat, prefix + "traj": pred_traj}


@MODULES.register_module()
class DDPMTrajCompletionFarEnvDecoder(DDPMTrajCompletionEnvDecoder):
    def _build_encoder_network(self):
        super()._build_encoder_network()
        self.envfarNet = nn.Sequential(
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
        self.IfarNet = nn.Sequential(
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
        self.far2rootNet = nn.Sequential(
            MLPBlock(
                self.root_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )
        self.far2goalNet = nn.Sequential(
            MLPBlock(
                self.root_dim,
                256,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, self.latent_dim, norm=self.norm, activation=self.activation),
        )

    def extract_feature(
        self,
        start_state,
        end_state,
        ItoR,
        ItoG,
        env,
        goal,
        root,
        Ifar,
        envfar,
        far2root,
        far2goal,
        pre_state,
        aft_state,
        *args,
        **kwargs,
    ):
        start_state = self.bistateNet(start_state)
        end_state = self.bistateNet(end_state)
        ItoR = self.ItoRNet(ItoR)
        ItoG = self.ItoGNet(ItoG)
        env = self.envNet(env)
        goal = self.goalNet(goal)
        root = self.rootNet(root)
        Ifar = self.IfarNet(Ifar)
        envfar = self.envfarNet(envfar)
        far2root = self.far2rootNet(far2root)
        far2goal = self.far2rootNet(far2goal)
        pre_state = self.bistateNet(pre_state)
        aft_state = self.bistateNet(aft_state)

        # conditions = torch.cat((start_state, end_state, ItoR, ItoG, root, goal), dim=1)
        conditions = torch.cat(
            (
                start_state,
                end_state,
                ItoR,
                ItoG,
                root,
                goal,
                env,
                Ifar,
                envfar,
                far2root,
                far2goal,
                pre_state,
                aft_state,
            ),
            dim=1,
        )
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(
        self,
        x,
        t,
        x_self_cond,
        start_state,
        end_state,
        ItoR,
        ItoG,
        env,
        goal,
        root,
        Ifar,
        envfar,
        far2root,
        far2goal,
        pre_state,
        aft_state,
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
        conditions = self.extract_feature(
            start_state,
            end_state,
            ItoR,
            ItoG,
            env,
            goal,
            root,
            Ifar,
            envfar,
            far2root,
            far2goal,
            pre_state,
            aft_state,
        )

        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim=-1)
        x = self.init_layer(x)  # b, l, c
        res = x.clone()
        x = x.permute(1, 0, 2)  # l, b, c

        time_embed = self.sequence_pos_encoding.query(t)
        time_embed = self.time_mlp(time_embed)  # b, 1, c
        time_embed = time_embed.permute(1, 0, 2)  # 1, b, c

        xseq = torch.cat((conditions, x), dim=0)
        xseq = self.sequence_pos_encoding(xseq)

        output = self.seqTransDecoder(tgt=xseq, memory=time_embed)
        output = output.permute(1, 0, 2)  # b, l, c
        extra_num = conditions.shape[0]
        output = output[:, extra_num:]
        output = torch.cat((res, output), dim=-1)
        output = self.res_layer(output, time_embed.permute(1, 0, 2))
        pred = self.predlayer(output)

        return {"pred": pred}


@MODULES.register_module()
class DDPMMilestonePoseEnvDecoder(ActorMilestonePoseEnvDecoder):
    def __init__(
        self,
        self_condition: bool = True,
        cond_mask_prob: float = 0.1,
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
        self.self_condition = self_condition
        self.cond_mask_prob = cond_mask_prob
        super().__init__(
            state_dim,
            input_pose_dim,
            env_dim,
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

    # def _build_transformer(self):
    #     self._build_tokens()
    #     if self.attention_op == "MHA":
    #         attention_op = None
    #     elif self.attention_op == "MHA":
    #         attention_op = MHA
    #     else:
    #         raise NotImplementedError

    #     seq_trans_decoder_layer = TransformerDecoderTimeEmbedLayer(
    #         d_model=self.latent_dim,
    #         nhead=self.num_heads,
    #         dim_feedforward=self.ff_size,
    #         dropout=self.dropout,
    #         attention_op=attention_op,
    #     )

    #     self.seqTransDecoder = nn.TransformerDecoder(
    #         seq_trans_decoder_layer, num_layers=self.num_layers
    #     )

    def _build_encoder_network(self):
        super()._build_encoder_network()
        input_channels = self.state_dim
        if self.self_condition:
            input_channels = input_channels * 2
        self.init_layer = nn.Linear(input_channels, self.latent_dim)
        self.time_mlp = nn.Sequential(
            MLPBlock(
                self.latent_dim,
                self.latent_dim,
                norm=self.norm,
                activation=self.activation,
            ),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def _build_final_layer(self):
        super()._build_final_layer()
        self.res_layer = ResTimeMLPBlock(
            self.latent_dim * 2,
            self.latent_dim,
            self.latent_dim,
            norm=self.norm,
        )

    def extract_feature(self, pose, I, env, traj, uncond=False, *args, **kwargs):
        pose = self.poseNet(pose)  # [B, 1, C]
        I = self.INet(I)  # [B, L, C]
        env = self.envNet(env)
        traj = self.trajNet(traj)  # [B, L, C]
        x = I + traj + env
        conditions = torch.cat((pose, x), dim=1)
        if uncond:
            conditions = torch.zeros_like(conditions)
        elif self.training and self.cond_mask_prob > 0.0:
            bs = conditions.shape[0]
            mask = torch.bernoulli(
                torch.ones(bs, device=conditions.device) * self.cond_mask_prob
            ).view(bs, 1, 1)
            conditions = conditions * (1.0 - mask)
        else:
            pass
        conditions = conditions.permute(1, 0, 2)  # l, b, c
        return conditions

    def forward(
        self,
        x,
        t,
        x_self_cond,
        pose,
        I,
        env,
        traj,
        lengths=None,
        mask=None,
        uncond=False,
        *args,
        **kwargs,
    ):
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
        traj_num = traj.shape[1]
        conditions = self.extract_feature(pose, I, env, traj, uncond=uncond)
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim=-1)
        x = self.init_layer(x)  # b, l, c
        res = x.clone()
        x = x.permute(1, 0, 2)  # l, b, c

        time_embed = self.sequence_pos_encoding.query(t)
        time_embed = self.time_mlp(time_embed)  # b, 1, c
        time_embed = time_embed.permute(1, 0, 2)  # 1, b, c

        conditions[-traj_num:] += x

        if lengths is None:
            lengths = [len(I_) for I_ in I]
        if mask is None:
            mask = lengths_to_mask(lengths, I.device)
        token_mask = torch.ones(pose.shape[:2], dtype=bool, device=x.device)
        aug_mask = torch.cat((token_mask, mask), 1)
        xseq = conditions
        # Pass through the transformer decoder
        # with the latent vector for memory
        xseq = self.sequence_pos_encoding(xseq)

        output = self.seqTransDecoder(
            tgt=xseq, memory=time_embed, tgt_key_padding_mask=~aug_mask
        )
        # zero for padded area
        output = output.permute(1, 0, 2)
        output = output[:, -traj_num:]
        output = torch.cat((res, output), dim=-1)
        output = self.res_layer(output, time_embed.permute(1, 0, 2))
        pred = self.pose_predlayer(output)
        return {"y_hat": pred, "pred": pred}

    def get_shape(self, data):
        B, L, _ = data["traj"].shape
        shape = (B, L, self.state_dim)
        return shape

    def inference_forward(self, *args, **kwargs):
        return self(*args, **kwargs)
