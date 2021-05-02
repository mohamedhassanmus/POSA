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

import torch
from src import data_utils


# Training dataset
class NPYDataset():
    def __init__(self, no_obj_classes=42, **kwargs):
        self.no_obj_classes = no_obj_classes
        self.x, _, _, self.vertices_can, self.x_semantics, _, _ = data_utils.load_data(**kwargs)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        item = {'x': self.x[idx].reshape(-1, 1), 'vertices_can': self.vertices_can[idx].reshape(-1, 3)}
        if self.x_semantics is not None:
            x_semantics = self.x_semantics[idx]
            x_semantics = torch.zeros(x_semantics.shape[0], self.no_obj_classes, dtype=torch.float32).scatter_(-1,
                                                                                                               x_semantics.unsqueeze(
                                                                                                                   -1).type(
                                                                                                                   torch.long),
                                                                                                               1.)
            item['x_semantics'] = x_semantics
        return item
