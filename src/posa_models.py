# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import torch
import torch.utils.data
from torch import nn
import numpy as np
import trimesh
import openmesh as om
from src import posa_utils


def load_model(model_name='POSA', **kwargs):
    if model_name == 'POSA':
        model = POSA
    else:
        err_msg = 'Unknown model name: {}'.format(model_name)
        raise ValueError(err_msg)

    output = model(**kwargs)
    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


## Common
def get_norm_layer(channels=None, normalization_mode=None, num_groups=None, affine=True):
    if num_groups is None:
        num_groups = num_groups
    if channels is None:
        channels = channels
    if normalization_mode is None:
        normalization_mode = normalization_mode
    if normalization_mode == 'batch_norm':
        return nn.BatchNorm1d(channels)
    elif normalization_mode == 'instance_norm':
        return (nn.InstanceNorm1d(channels))
    elif normalization_mode == 'layer_norm':
        return (nn.LayerNorm(channels))
    elif normalization_mode == 'group_norm':
        return (nn.GroupNorm(num_groups, channels, affine=affine))


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))

        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


class GraphLin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphLin, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        x = self.layer(x)
        return x


class GraphLin_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalization_mode=None, num_groups=None,
                 drop_out=False, non_lin=True):
        super(GraphLin_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization_mode = normalization_mode
        self.non_lin = non_lin
        self.drop_out = drop_out

        self.conv = GraphLin(in_channels, out_channels)
        if self.normalization_mode is not None:
            if self.out_channels % num_groups != 0:
                num_groups = self.out_channels
            self.norm = get_norm_layer(self.out_channels, normalization_mode, num_groups)
        if self.non_lin:
            self.relu = nn.ReLU()
        if self.drop_out:
            self.drop_out_layer = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        if self.normalization_mode is not None:
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.non_lin:
            x = self.relu(x)
        if self.drop_out:
            x = self.drop_out_layer(x)
        return x


class Spiral_block(nn.Module):
    def __init__(self, in_channels, out_channels, indices, normalization_mode=None, num_groups=None,
                 non_lin=True):
        super(Spiral_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization_mode = normalization_mode
        self.non_lin = non_lin

        self.conv = SpiralConv(in_channels, out_channels, indices)
        if self.normalization_mode is not None:
            if self.out_channels % num_groups != 0:
                num_groups = self.out_channels
            self.norm = get_norm_layer(self.out_channels, normalization_mode, num_groups)
        if self.non_lin:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.normalization_mode is not None:
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.non_lin:
            x = self.relu(x)
        return x


class fc_block(nn.Module):
    def __init__(self, in_features, out_features, normalization_mode=None, drop_out=False, non_lin=True):
        super(fc_block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalization_mode = normalization_mode
        self.non_lin = non_lin
        self.drop_out = drop_out

        self.lin = nn.Linear(in_features, out_features)
        if self.normalization_mode is not None:
            self.norm = get_norm_layer(self.out_features, self.normalization_mode)

        if self.non_lin:
            self.relu = nn.ReLU()
        if self.drop_out:
            self.drop_out_layer = nn.Dropout(0.5)

    def forward(self, x):
        x = self.lin(x)
        if self.normalization_mode is not None:
            x = self.norm(x)
        if self.non_lin:
            x = self.relu(x)

        return x


class ds_us_fn(nn.Module):
    def __init__(self, M):
        super(ds_us_fn, self).__init__()
        self.M = M

    def forward(self, x):
        return torch.matmul(self.M, x)


def load_ds_us_param(ds_us_dir, level, seq_length, use_cuda=True):
    ds_us_dir = osp.abspath(ds_us_dir)
    device = torch.device("cuda" if use_cuda else "cpu")
    level = level + 2  # Start from 2

    m = trimesh.load(osp.join(ds_us_dir, 'mesh_{}.obj'.format(level)), process=False)
    spiral_indices = torch.tensor(posa_utils.extract_spirals(om.TriMesh(m.vertices, m.faces), seq_length)).to(device)
    nv = m.vertices.shape[0]
    verts_T_pose = torch.tensor(m.vertices, dtype=torch.float32).to(device)

    A, U, D = posa_utils.get_graph_params(ds_us_dir, level, use_cuda=use_cuda)
    A = A.to_dense()
    U = U.to_dense()
    D = D.to_dense()
    return nv, spiral_indices, A, U, D, verts_T_pose


class Encoder(nn.Module):
    def __init__(self, h_dim=512, z_dim=256, channels=64, ds_us_dir='./mesh_ds', normalization_mode='group_norm',
                 num_groups=8, seq_length=9,
                 use_semantics=False, no_obj_classes=42,
                 use_cuda=True, **kwargs):
        super(Encoder, self).__init__()

        self.f_dim = 1
        if use_semantics:
            self.f_dim += no_obj_classes
        self.spiral_indices = []
        self.nv = []
        self.D = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, _, D, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)
            self.D.append(D)
            self.spiral_indices.append(spiral_indices)
        self.channels = (channels * np.ones(4)).astype(np.int).tolist()

        self.en_spiral = nn.ModuleList()
        self.en_spiral.append(
            Spiral_block(3 + self.f_dim, self.channels[0], self.spiral_indices[0], normalization_mode, num_groups))
        for i in levels:
            self.en_spiral.append(
                Spiral_block(self.channels[i], self.channels[i + 1], self.spiral_indices[i], normalization_mode,
                             num_groups))
            if i != len(levels) - 1:
                self.en_spiral.append(ds_us_fn(self.D[i + 1]))

        self.en_spiral = nn.Sequential(*self.en_spiral)

        self.en_fc = nn.ModuleList()
        self.en_fc.append(fc_block(self.nv[-1] * self.channels[-1], h_dim, normalization_mode='layer_norm'))
        self.en_fc = nn.Sequential(*self.en_fc)
        self.en_mu = nn.Linear(h_dim, z_dim)
        self.en_log_var = nn.Linear(h_dim, z_dim)

    def forward(self, x, vertices):
        x = torch.cat((vertices, x), dim=-1)
        x = self.en_spiral(x)
        x = x.reshape(-1, self.nv[-1] * self.channels[-1])
        x = self.en_fc(x)
        return self.en_mu(x), self.en_log_var(x)


class Decoder(nn.Module):
    def __init__(self, z_dim=256, num_hidden_layers=3, channels=64, ds_us_dir='./mesh_ds',
                 normalization_mode='group_norm', num_groups=8, seq_length=9,
                 use_semantics=False, no_obj_classes=42,
                 use_cuda=True, **kwargs):
        super(Decoder, self).__init__()
        self.f_dim = 1
        self.use_semantics = use_semantics
        if self.use_semantics:
            self.f_dim += no_obj_classes
        self.spiral_indices = []
        self.nv = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, _, _, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)
            self.spiral_indices.append(spiral_indices)
        self.channels = (channels * np.ones(4)).astype(np.int).tolist()

        self.de_spiral = nn.ModuleList()
        self.de_spiral.append(GraphLin_block(3 + z_dim, z_dim // 2, normalization_mode, num_groups))
        self.de_spiral.append(GraphLin_block(z_dim // 2, self.channels[0], normalization_mode, num_groups))
        for _ in range(num_hidden_layers):
            self.de_spiral.append(
                Spiral_block(self.channels[0], self.channels[0], self.spiral_indices[0], normalization_mode,
                             num_groups))
        self.de_spiral.append(SpiralConv(self.channels[0], self.f_dim, self.spiral_indices[0]))
        self.de_spiral = nn.Sequential(*self.de_spiral)

    def forward(self, x, vertices):
        x = x.unsqueeze(1).expand((-1, self.nv[0], -1))
        x = torch.cat((vertices, x), dim=-1)
        x = self.de_spiral(x)

        x_d = torch.sigmoid(x[:, :, 0]).unsqueeze(-1)
        out = x_d
        if self.use_semantics:
            x_semantics = x[:, :, 1:]
            out = torch.cat((out, x_semantics), dim=-1)
        return out


class POSA(nn.Module):
    def __init__(self, **kwargs):
        super(POSA, self).__init__()
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, vertices=None):
        mu, logvar = self.encoder(x, vertices)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z, vertices)
        return out, mu, logvar
