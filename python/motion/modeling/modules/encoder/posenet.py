import torch
import torch.nn as nn
import numpy as np
from motion.modeling.operators.activation import ACTIVATIONS
from motion.modeling.modules.builder import MODULES
from motion.modeling.operators.mlp import MLPBlock, Conv1dBlock


@MODULES.register_module()
class ContactEnvPoseNetEncoder(nn.Module):
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
        self.contact_encoder = nn.Sequential(
            MLPBlock(
                self.contact_dim, 256, dropout=self.dropout, activation=self.activation
            ),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
        )

        self.main = nn.Sequential(
            MLPBlock(
                self.pose_dim, 256, dropout=self.dropout, activation=self.activation
            ),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
        )

        self.fc1 = nn.Linear(1024, self.z_dim)
        self.fc2 = nn.Linear(1024, self.z_dim)

    def forward(self, y, env, action, contact, *args, **kwargs):
        env = self.env_encoder(env)
        action = self.action_encoder(action)
        contact = self.contact_encoder(contact)
        x = self.main(y)
        x = torch.cat((x, contact, env, action), dim=-1)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return {"mu": mu, "var": logvar.exp(), "logvar": logvar}


@MODULES.register_module()
class VQContactEnvPosePartEncoder(ContactEnvPoseNetEncoder):
    def __init__(
        self,
        joint_dim,
        quant_dim_enc,
        pose_dim,
        env_dim,
        contact_dim,
        action_dim,
        z_dim,
        dropout,
        activation,
        part_dict,
        **kwargs,
    ):
        self.joint_dim = joint_dim
        self.quant_dim_enc = quant_dim_enc
        self.part_dict = part_dict

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
        for k, v in self.part_dict.items():
            fc = nn.Sequential(
                MLPBlock(
                    len(v) * self.joint_dim,
                    256,
                    dropout=self.dropout,
                    activation=self.activation,
                ),
                MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
                MLPBlock(256, self.quant_dim_enc),
            )
            setattr(self, f"fc{k}", fc)

        self.contact_encoder = nn.Sequential(
            MLPBlock(
                self.contact_dim, 256, dropout=self.dropout, activation=self.activation
            ),
            MLPBlock(256, 256, dropout=self.dropout, activation=self.activation),
            MLPBlock(256, self.quant_dim_enc),
        )

    def forward(self, y, contact, *args, **kwargs):
        contact = self.contact_encoder(contact)
        feats = {"contact": contact}
        y = y.reshape(y.shape[:-1] + (-1, self.joint_dim))
        for k, v in self.part_dict.items():
            part_y = y[..., v, :].flatten(-2)
            fc_layer = getattr(self, f"fc{k}")
            feats[k] = fc_layer(part_y)
        if len(self.part_dict.keys()) == 1:
            feats[list(self.part_dict.keys())[0]] += feats["contact"]
        return feats
