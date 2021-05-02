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
import torch.nn.functional as F
import numpy as np
import torchgeometry as tgm
from src import misc_utils
import src.posa_models as models
import smplx
import cv2


def read_sdf(vertices, sdf_grid, grid_dim, grid_min, grid_max, mode='bilinear'):
    assert vertices.dim() == 3
    assert sdf_grid.dim() == 4
    # sdf_normals: B*dim*dim*dim*3
    batch_size = vertices.shape[0]
    nv = vertices.shape[1]
    sdf_grid = sdf_grid.unsqueeze(0).permute(0, 4, 1, 2, 3)  # B*C*D*D*D
    norm_vertices = (vertices - grid_min) / (grid_max - grid_min) * 2 - 1
    x = F.grid_sample(sdf_grid,
                      norm_vertices[:, :, [2, 1, 0]].view(batch_size, nv, 1, 1, 3),
                      padding_mode='border', mode=mode, align_corners=True)
    x = x.permute(0, 2, 3, 4, 1)
    return x


def smpl_in_new_coords(torch_param, Rcw, Tcw, rotation_center, **kwargs):
    body_model = misc_utils.load_body_model(**kwargs).to(Rcw.device)
    body_model.reset_params(betas=torch_param['betas'])
    #
    P = body_model().joints.detach().squeeze()[0, :].reshape(-1, 3)
    global_orient_c = torch_param['global_orient']
    Tc = torch_param['transl']

    Rc = tgm.angle_axis_to_rotation_matrix(global_orient_c.reshape(-1, 3))[:, :3, :3]
    Rw = torch.matmul(Rcw, Rc)
    global_orient_w = cv2.Rodrigues(Rw.detach().cpu().squeeze().numpy())[0]
    torch_param['global_orient'] = torch.tensor(global_orient_w, dtype=global_orient_c.dtype,
                                                device=global_orient_c.device).reshape(1, 3)

    torch_param['transl'] = torch.matmul(Rcw, (P + Tc - rotation_center).t()).t() + Tcw - P
    return torch_param


def compute_recon_loss(gt_batch, pr_batch, contact_w, semantics_w, use_semantics, loss_type, reduction='mean',
                       **kwargs):
    batch_size = gt_batch.shape[0]
    device = gt_batch.device
    dtype = gt_batch.dtype
    recon_loss_dist = torch.zeros(1, dtype=dtype, device=device)
    recon_loss_semantics = torch.zeros(1, dtype=dtype, device=device)
    semantics_recon_acc = torch.zeros(1, dtype=dtype, device=device)
    if loss_type == 'bce':
        recon_loss_dist = contact_w * F.binary_cross_entropy(pr_batch[:, :, 0], gt_batch[:, :, 0], reduction=reduction)
    elif loss_type == 'mse':
        recon_loss_dist = contact_w * F.mse_loss(pr_batch[:, :, 0], gt_batch[:, :, 0], reduction=reduction)
    if use_semantics:
        targets = gt_batch[:, :, 1:].argmax(dim=-1).type(torch.long).reshape(batch_size, -1)
        recon_loss_semantics = semantics_w * F.cross_entropy(pr_batch[:, :, 1:].permute(0, 2, 1), targets,
                                                             reduction=reduction)

        semantics_recon_acc = torch.mean((targets == torch.argmax(pr_batch[:, :, 1:], dim=-1)).float())

    return recon_loss_dist, recon_loss_semantics, semantics_recon_acc


def rotmat2transmat(x):
    x = np.append(x, np.array([0, 0, 0]).reshape(1, 3), axis=0)
    x = np.append(x, np.array([0, 0, 0, 1]).reshape(4, 1), axis=1)
    return x


def create_init_points(bbox, mesh_grid_step, pelvis_z_offset=0.0):
    x_offset = 0.75
    y_offset = 0.75
    X, Y, Z = np.meshgrid(np.arange(bbox[1, 0] + x_offset, bbox[0, 0] - x_offset, mesh_grid_step),
                          np.arange(bbox[1, 1] + y_offset, bbox[0, 1] - y_offset, mesh_grid_step),
                          np.arange(bbox[1, 2] + pelvis_z_offset, bbox[1, 2] + pelvis_z_offset + 0.5, mesh_grid_step))
    init_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
    return init_points


def load_body_model(model_folder, num_pca_comps=6, batch_size=1, gender='male', **kwargs):
    model_params = dict(model_path=model_folder,
                        model_type='smplx',
                        ext='npz',
                        num_pca_comps=num_pca_comps,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        batch_size=batch_size)

    body_model = smplx.create(gender=gender, **model_params)
    return body_model


def load_model_checkpoint(device, model_name='POSA', load_checkpoint=None, use_cuda=False, checkpoints_dir=None,
                          checkpoint_path=None, **kwargs):
    model = models.load_model(model_name, use_cuda=use_cuda, **kwargs)
    model.eval()
    if checkpoint_path is not None:
        print('loading {}'.format(checkpoint_path))
        if not use_cuda:
            checkpoint = torch.load(osp.join(checkpoint_path),
                                    map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    elif load_checkpoint > 0:
        print('loading stats of epoch {} from {}'.format(load_checkpoint, checkpoints_dir))
        if not use_cuda:
            checkpoint = torch.load(osp.join(checkpoints_dir, 'epoch_{:04d}.pt'.format(load_checkpoint)),
                                    map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(osp.join(checkpoints_dir, 'epoch_{:04d}.pt'.format(load_checkpoint)))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No checkpoint found')
        sys.exit(0)
    return model


def concat_resplit(x1, x2, ind1, ind2):
    if torch.is_tensor(x1):
        x = torch.cat((x1, x2))
    else:
        x = np.concatenate((x1, x2))
    x1 = x[ind1]
    x2 = x[ind2]
    return x1, x2


def rot_mat_to_euler(rot_mats):
    # order of rotation zyx
    R_y = -torch.asin(rot_mats[:, 2, 0])
    R_x = torch.atan2(rot_mats[:, 2, 1] / torch.cos(R_y), rot_mats[:, 2, 2] / torch.cos(R_y))
    R_z = torch.atan2(rot_mats[:, 1, 0] / torch.cos(R_y), rot_mats[:, 0, 0] / torch.cos(R_y))
    return R_z, R_y, R_x


def eval_physical_metric(vertices, scene_data):
    nv = float(vertices.shape[1])
    x = misc_utils.read_sdf(vertices, scene_data['sdf'],
                            scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                            mode='bilinear').squeeze()

    if x.lt(0).sum().item() < 1:  # if the number of negative sdf entries is less than one
        non_collision_score = torch.tensor(1)
        contact_score = torch.tensor(0.0)
    else:
        non_collision_score = (x > 0).sum().float() / nv
        contact_score = torch.tensor(1.0)

    return float(non_collision_score.detach().cpu().squeeze()), float(contact_score.detach().cpu().squeeze())
