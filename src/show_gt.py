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
import open3d as o3d
import torch
import trimesh
import random
from src.cmd_parser import parse_config
from src import data_utils, posa_utils, viz_utils

if __name__ == '__main__':
    args, args_dict = parse_config()
    args_dict['batch_size'] = 1
    args_dict['use_cuda'] = 0
    args_dict['base_dir'] = osp.expandvars(args_dict.get('base_dir'))
    args_dict['data_dir'] = osp.expandvars(args_dict.get('data_dir'))
    args_dict['ds_us_dir'] = osp.expandvars(args_dict.get('ds_us_dir'))
    base_dir = args_dict.get('base_dir')
    ds_us_dir = args_dict.get('ds_us_dir')
    device = torch.device("cpu")
    dtype = torch.float32

    A_1, U_1, D_1 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 1, args_dict['use_cuda'])
    down_sample_fn = posa_utils.ds_us(D_1).to(device)
    up_sample_fn = posa_utils.ds_us(U_1).to(device)

    A_2, U_2, D_2 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 2, args_dict['use_cuda'])
    down_sample_fn2 = posa_utils.ds_us(D_2).to(device)
    up_sample_fn2 = posa_utils.ds_us(U_2).to(device)

    faces_arr = trimesh.load(osp.join(ds_us_dir, 'mesh_{}.obj'.format(args_dict.get('down_sample'))),
                             process=False).faces
    if args.up_sample:
        faces_arr = trimesh.load(osp.join(ds_us_dir, 'mesh_{}.obj'.format(0)), process=False).faces

    x, joints_can, vertices, vertices_can, x_semantics, recording_names, pkl_file_paths = data_utils.load_data(
        **args_dict)

    print("Number of examples is {}".format(len(x)))
    ind = list(range(len(x)))
    if args.shuffle:
        random.shuffle(ind)
    for i in ind:
        recording_name = recording_names[i]
        pkl_file_path = pkl_file_paths[i]
        scene_name = recording_name.split('_')[0]
        print('viewing frame {} of {}'.format(pkl_file_path, recording_name))

        results = []
        scene = o3d.io.read_triangle_mesh(osp.join(base_dir, 'scenes', scene_name + '.ply'))
        results += [scene]
        x_i = x[i].view(args.batch_size, args.nv, 1)
        in_batch = x_i
        vertices_i = vertices[i]
        if args.use_semantics:
            x_semantics_i = x_semantics[i].reshape(args.batch_size, args.nv)
            x_semantics_i = torch.zeros(x_semantics_i.shape[0], x_semantics_i.shape[1], args.no_obj_classes,
                                        dtype=dtype, device=device).scatter_(-1, x_semantics_i.unsqueeze(-1).type(
                torch.long), 1.)
        in_batch = torch.cat((in_batch, x_semantics_i), dim=-1)
        if args.up_sample:
            vertices_i = up_sample_fn2.forward(vertices_i.unsqueeze(0).permute(0, 2, 1))
            vertices_i = up_sample_fn.forward(vertices_i).permute(0, 2, 1).squeeze()

            in_batch = in_batch.transpose(1, 2)
            in_batch = up_sample_fn2.forward(in_batch)
            in_batch = up_sample_fn.forward(in_batch)
            in_batch = in_batch.transpose(1, 2)
        results += viz_utils.show_sample(vertices_i, in_batch, faces_arr, **args_dict)
        o3d.visualization.draw_geometries(results)
