import numpy as np
import torch
import torch.nn as nn

from motion.modeling.operators.mlp import MLPBlock, Conv1dBlock
from motion.modeling.modules.builder import MODULES


@MODULES.register_module()
class ContactEnvPoseNetDecoder(nn.Module):
    def __init__(
        self,
        pose_dim,
        env_dim,
        contact_dim,
        action_dim,
        z_dim,
        dropout,
        activation,
        **kwargs,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.env_dim = env_dim
        self.contact_dim = contact_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.dropout = dropout
        self.activation = activation
        self._build_model()

    def _build_model(self):
        self._build_condition()
        self._build_posenet()

    def _build_condition(self):
        self.action_encoder = nn.Linear(self.action_dim, 256)

        self.env_encoder = nn.Sequential(
            MLPBlock(
                self.env_dim, 256, dropout=self.dropout, activation=self.activation
            ),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
        )

    def _build_posenet(self):
        self.fc1 = MLPBlock(
            256 * 2 + self.z_dim, 256, dropout=self.dropout, activation=self.activation
        )
        self.fc2 = MLPBlock(
            256 + self.z_dim, 256, dropout=self.dropout, activation=self.activation
        )

        self.fc_pose = MLPBlock(256 + self.z_dim, self.pose_dim)
        self.fc_contact = MLPBlock(256 + self.z_dim, self.contact_dim)

    def forward(self, z, env, action, *args, **kwargs):
        env = self.env_encoder(env)
        action = self.action_encoder(action)
        x = torch.cat((env, action, z), dim=-1)
        x = self.fc1(x)
        x = torch.cat((x, z), dim=-1)
        x = self.fc2(x)
        x = torch.cat((x, z), dim=-1)
        y_hat = self.fc_pose(x)
        y_contact = self.fc_contact(x)
        return {"y_hat": y_hat, "contact": y_contact}

    def inference_forward(self, *args, **kwargs):
        return self(*args, **kwargs)


@MODULES.register_module()
class VQContactEnvPosePartDecoder(ContactEnvPoseNetDecoder):
    def __init__(
        self,
        part_n,
        quant_dim_dec,
        pose_dim,
        env_dim,
        contact_dim,
        action_dim,
        z_dim,
        dropout,
        activation,
        **kwargs,
    ):
        self.part_n = part_n
        self.quant_dim_dec = quant_dim_dec
        super().__init__(
            pose_dim,
            env_dim,
            contact_dim,
            action_dim,
            z_dim,
            dropout,
            activation,
            **kwargs,
        )

    def _build_condition(self):
        pass

    def _build_posenet(self):
        self.main = nn.Sequential(
            MLPBlock(
                self.quant_dim_dec,
                256,
                dropout=self.dropout,
                activation=self.activation,
            ),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
        )
        self.fc_pose = MLPBlock(256, self.pose_dim)
        self.fc_contact = MLPBlock(256, self.contact_dim)

    def forward(self, *args, **kwargs):
        quants = kwargs["quant"]
        quant_x = torch.cat([x for x in quants.values()], dim=-1)
        x = self.main(quant_x)
        y_hat = self.fc_pose(x)
        y_contact = self.fc_contact(x)
        return {"y_hat": y_hat, "contact": y_contact}
