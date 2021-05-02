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

import os
import os.path as osp
import numpy as np
import open3d as o3d
import torch
import trimesh
import glob
from src import viz_utils, misc_utils, posa_utils, data_utils
from src.cmd_parser import parse_config

if __name__ == '__main__':
    args, args_dict = parse_config()
    args_dict['batch_size'] = 1
    args_dict['ds_us_dir'] = osp.expandvars(args_dict.get('ds_us_dir'))
    args_dict['rand_samples_dir'] = osp.expandvars(args_dict.get('rand_samples_dir'))
    args_dict['model_folder'] = osp.expandvars(args_dict.get('model_folder'))

    ds_us_dir = args_dict.get('ds_us_dir')
    rand_samples_dir = args_dict.get('rand_samples_dir')
    os.makedirs(rand_samples_dir, exist_ok=True)

    device = torch.device("cuda" if args_dict.get('use_cuda') else "cpu")
    dtype = torch.float32

    A_1, U_1, D_1 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 1, args_dict['use_cuda'])
    down_sample_fn = posa_utils.ds_us(D_1).to(device)
    up_sample_fn = posa_utils.ds_us(U_1).to(device)

    A_2, U_2, D_2 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 2, args_dict['use_cuda'])
    down_sample_fn2 = posa_utils.ds_us(D_2).to(device)
    up_sample_fn2 = posa_utils.ds_us(U_2).to(device)

    faces_arr = trimesh.load(osp.join(ds_us_dir, 'mesh_{}.obj'.format(0)), process=False).faces

    model = misc_utils.load_model_checkpoint(device=device, **args_dict).to(device)

    pkl_file_path = args_dict.pop('pkl_file_path')
    if osp.isdir(pkl_file_path):
        pkl_file_dir = pkl_file_path
        pkl_file_paths = glob.glob(osp.join(pkl_file_dir, '*.pkl'))
    else:
        pkl_file_paths = [pkl_file_path]

    for pkl_file_path in pkl_file_paths:
        print('file_name: {}'.format(pkl_file_path))
        pkl_file_basename = osp.splitext(osp.basename(pkl_file_path))[0]

        # load pkl file
        vertices, vertices_can, faces_arr, body_model, R_can, pelvis, torch_param, _ = data_utils.pkl_to_canonical(
            pkl_file_path, device, dtype, **args_dict)

        vertices_can_ds = down_sample_fn.forward(vertices_can.unsqueeze(0).permute(0, 2, 1))
        vertices_can_ds = down_sample_fn2.forward(vertices_can_ds).permute(0, 2, 1).squeeze()

        z = torch.tensor(np.random.normal(0, 1, (args.num_rand_samples, args.z_dim)).astype(np.float32)).to(
            device)
        gen_batch = model.decoder(z, vertices_can_ds.expand(args.num_rand_samples, -1, -1))

        gen_batch = gen_batch.transpose(1, 2)
        gen_batch = up_sample_fn2.forward(gen_batch)
        gen_batch = up_sample_fn.forward(gen_batch)
        gen_batch = gen_batch.transpose(1, 2)

        if args.viz:
            results = []
            for i in range(args.num_rand_samples):
                gen = viz_utils.show_sample(vertices_can, gen_batch[i], faces_arr, **args_dict)
                for m in gen:
                    trans = np.eye(4)
                    trans[1, 3] = 2 * i
                    m.transform(trans)
                    results.append(m)
            o3d.visualization.draw_geometries(results)

        if args.render:
            gen_batch = gen_batch.detach().cpu().numpy()
            img = viz_utils.render_sample(gen_batch, vertices, faces_arr, **args_dict)
            for i in range(args.num_rand_samples):
                img[i].save(osp.join(rand_samples_dir, '{}_sample_{:02d}.png'.format(pkl_file_basename, i)))
