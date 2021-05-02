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
import numpy as np
import pandas as pd
import trimesh
import pyrender
import PIL.Image as pil_img
from src import misc_utils, data_utils, eulerangles

default_color = [1.00, 0.75, 0.80]
view_rotations = [misc_utils.rotmat2transmat(eulerangles.euler2mat(0, 0, 0, 'szyx')),
                  misc_utils.rotmat2transmat(eulerangles.euler2mat(np.pi / 2, 0, 0, 'szyx')),
                  misc_utils.rotmat2transmat(eulerangles.euler2mat(np.pi, 0, 0, 'szyx')),
                  misc_utils.rotmat2transmat(eulerangles.euler2mat(np.pi / 2, np.pi, 0, 'sxzy'))]


def create_renderer(H=1080, W=1920, intensity=50, fov=None):
    if fov is None:
        fov = np.pi / 3.0
    r = pyrender.OffscreenRenderer(viewport_width=W,
                                   viewport_height=H)
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.333)
    light_directional = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
    light_point = pyrender.PointLight(color=np.ones(3), intensity=intensity)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    return r, camera, light_directional, light_point, material


def create_pyrender_scene(camera, camera_pose, light_directional=None, light_point=None, light_pose=None):
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=camera_pose)
    if light_pose is None:
        light_pose = camera_pose
    if light_directional is not None:
        scene.add(light_directional, pose=np.eye(4))
    if light_point is not None:
        scene.add(light_point, pose=light_pose)
    return scene


def render_body(scene, renderer, body_trimesh=None, vertices=None, faces_arr=None, vertex_colors=None, material=None):
    if material is None:
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    if isinstance(body_trimesh, list):
        for body in body_trimesh:
            body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
            scene.add(body_mesh, 'mesh')
    elif body_trimesh is None:
        body_trimesh = trimesh.Trimesh(vertices, faces_arr, vertex_colors=vertex_colors * 255,
                                       process=False)
        body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)
        scene.add(body_mesh, 'mesh')

    else:
        body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)
        scene.add(body_mesh, 'mesh')

    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    return color


def create_collage(images, mode='grid'):
    n = len(images)
    W, H = images[0].size
    if mode == 'grid':
        img_collage = pil_img.new('RGB', (2 * W, 2 * H))
        for id, img in enumerate(images):
            img_collage.paste(img, (W * (id % 2), H * int(id / 2)))
    elif mode == 'vertical':
        img_collage = pil_img.new('RGB', (W, n * H))
        for id, img in enumerate(images):
            img_collage.paste(img, (0, id * H))
    elif mode == 'horizantal':
        img_collage = pil_img.new('RGB', (n * W, H))
        for id, img in enumerate(images):
            img_collage.paste(img, (id * W, 0))
    return img_collage


def render_interaction_snapshot(body, static_scene, clothed_body, use_clothed_mesh=False, body_center=True,
                                collage_mode='grid', **kwargs):
    H, W = 480, 640
    renderer, camera, light_directional, light_point, material = create_renderer(H=H, W=W, intensity=2.0)
    light_point.intensity = 10.0

    # this will make the camera looks in the -x direction
    camera_pose = np.eye(4)
    camera_pose[0, 3] = 2
    camera_pose[2, 3] = 1
    camera_pose[:3, :3] = eulerangles.euler2mat(-np.pi / 6, np.pi / 2, np.pi / 2, 'sxzy')

    if body_center:
        center = (body.vertices.max(axis=0) + body.vertices.min(axis=0)) / 2.0
    else:
        center = (static_scene.vertices.max(axis=0) + static_scene.vertices.min(axis=0)) / 2.0
        camera_pose[0, 3] = 3

    static_scene.vertices -= center
    body.vertices -= center
    if use_clothed_mesh:
        clothed_body.vertices -= center

    images = []
    # for ang_id, ang in enumerate(range(0, 360, 90)):
    for ang_id, ang in enumerate([0, 90]):
        ang = np.pi / 180 * ang
        rot_z = np.eye(4)
        rot_z[:3, :3] = eulerangles.euler2mat(ang, 0, 0, 'szxy')

        static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)
        if use_clothed_mesh:
            body_mesh = pyrender.Mesh.from_trimesh(clothed_body, material=material)
        else:
            body_mesh = pyrender.Mesh.from_trimesh(body, material=material)

        scene = pyrender.Scene()
        scene.add(camera, pose=np.matmul(rot_z, camera_pose))
        scene.add(light_point, pose=np.matmul(rot_z, camera_pose))
        scene.add(light_directional, pose=np.eye(4))
        scene.add(static_scene_mesh, 'mesh')
        scene.add(body_mesh, 'mesh')

        color, _ = renderer.render(scene, pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL)
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        images.append(img)
    images = create_collage(images, collage_mode)
    static_scene.vertices += center
    return images


def render_multi_view(vertices, faces_arr, vertex_colors, camera_pose, renderer, camera, light_directional, light_point,
                      material, num_views=4):
    images = []
    for view_id, rot in enumerate(view_rotations[:num_views]):
        scene = create_pyrender_scene(camera, np.matmul(rot, camera_pose), light_directional, light_point)
        color = render_body(scene, renderer, vertices=vertices, faces_arr=faces_arr, vertex_colors=vertex_colors,
                            material=material)
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        images.append(img)
    return images


def render_sample(in_batch, vertices, faces_arr, use_semantics=False, make_collage=True, **kwargs):
    batch_size, nv, nf = in_batch.shape

    H, W = 480, 640
    renderer, camera, light_directional, light_point, material = create_renderer(H=H, W=W, intensity=2.5)
    images = {}
    vertices = vertices.squeeze()
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    vertices -= (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    camera_pose = np.eye(4)
    camera_pose[1, 3] = -1.5
    camera_pose[:3, :3] = eulerangles.euler2mat(0, 0, np.pi / 2, 'szyx')

    semantics_color_coding = get_semantics_color_coding()
    results = []
    for i in range(in_batch.shape[0]):
        x, x_semantics = data_utils.batch2features(in_batch[i].reshape(1, nv, nf), use_semantics)
        x_contact = (x > 0.5).astype(np.int)
        if use_semantics:
            x_semantics = np.argmax(x_semantics, axis=1)
        # contact
        vertex_colors = np.ones((vertices.shape[0], 3)) * np.array(default_color)
        vertex_colors[x_contact.flatten() == 1, :3] = [0.0, 0.0, 1.0]
        images_contact = render_multi_view(vertices, faces_arr, vertex_colors, camera_pose, renderer, camera,
                                           light_directional, light_point,
                                           material)
        images['contact'] = images_contact

        if use_semantics:
            vertex_colors = np.zeros((vertices.shape[0], 3))
            vertex_colors[:, :3] = default_color
            vertex_colors[x_semantics != 0, :3] = np.take(semantics_color_coding,
                                                          list(x_semantics[x_semantics != 0]),
                                                          axis=0) / 255.0

            images_semantics = render_multi_view(vertices, faces_arr, vertex_colors, camera_pose, renderer, camera,
                                                 light_directional, light_point,
                                                 material)
            images['semantics'] = images_semantics

        if make_collage:
            l = []
            for key in images.keys():
                l.append(create_collage(images[key], mode='horizantal'))
            img = create_collage(l, mode='vertical')
            results.append(img)
        else:
            results.append(images)

    return results


def composite_two_imgs(img1, img2):
    valid_mask = (img1[:, :, -1] > 0)[:, :, np.newaxis]
    output_img = (img1[:, :, :] * valid_mask +
                  (1 - valid_mask) * img2[:, :, :])
    return output_img


def create_o3d_mesh_from_np(vertices, faces, vertex_colors=[]):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.vertex_normals = o3d.utility.Vector3dVector([])
    mesh.triangle_normals = o3d.utility.Vector3dVector([])
    mesh.compute_vertex_normals()
    return mesh


def show_semantics_fn(vertices, x_semantics, faces_arr, **kwargs):
    semantics_color_coding = get_semantics_color_coding()
    if torch.is_tensor(x_semantics):
        x_semantics = x_semantics.detach().cpu().numpy().squeeze()
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy().squeeze()

    x_semantics = np.argmax(x_semantics, axis=1)
    vertex_colors = np.ones((vertices.shape[0], 3)) * np.array(default_color)

    vertex_colors[:, :3] = np.take(semantics_color_coding, list(x_semantics), axis=0) / 255.0
    vertex_colors[x_semantics == 0, :] = default_color

    body = create_o3d_mesh_from_np(vertices=vertices, faces=faces_arr,
                                   vertex_colors=vertex_colors)
    return [body]


def show_contact_fn(vertices, x, faces_arr, **kwargs):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().squeeze()

    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy().squeeze()

    x = (x > 0.5).astype(np.int)
    vertex_colors = np.ones((vertices.shape[0], 3)) * np.array(default_color)
    vertex_colors[x == 1, :3] = [0.0, 0.0, 1.0]
    body_gt_contact = create_o3d_mesh_from_np(vertices=vertices, faces=faces_arr,
                                              vertex_colors=vertex_colors)
    return [body_gt_contact]


def show_sample(vertices, in_batch, faces_arr, use_semantics, make_canonical=True, use_shift=True, **kwargs):
    results = []
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy().squeeze()
    if not make_canonical:
        vertex_colors = np.ones((vertices.shape[0], 3)) * np.array(default_color)
        body = create_o3d_mesh_from_np(vertices=vertices, faces=faces_arr,
                                       vertex_colors=vertex_colors)
        results.append(body)
        vertices = vertices + np.array([0, 0.0, 2.0])

    shift = 0
    x, x_semantics = data_utils.batch2features(in_batch, use_semantics, **kwargs)
    x_mesh = show_contact_fn(vertices, x, faces_arr, **kwargs)
    results += x_mesh
    if use_shift:
        shift += 2.0

    if use_semantics:
        if use_shift:
            shift += 0.0
        x_semantics_mesh = show_semantics_fn(vertices + np.array([0.0, 0.0, shift]).reshape(1, 3), x_semantics,
                                             faces_arr, **kwargs)
        results += x_semantics_mesh

    return results


def hex2rgb(hex_color_list):
    rgb_list = []
    for hex_color in hex_color_list:
        h = hex_color.lstrip('#')
        rgb = list(int(h[i:i + 2], 16) for i in (0, 2, 4))
        rgb_list.append(rgb)

    return np.array(rgb_list)


def get_semantics_color_coding():
    matter_port_label_filename = './mpcat40.tsv'
    matter_port_label_filename = osp.expandvars(matter_port_label_filename)
    df = pd.read_csv(matter_port_label_filename, sep='\t')
    color_coding_hex = list(df['hex'])  # list of str
    color_coding_rgb = hex2rgb(color_coding_hex)
    return color_coding_rgb


def create_o3d_sphere(pos, radius=0.01):
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sp.paint_uniform_color([1.0, 0.0, 0.0])
    T = np.eye(4)
    T[:3, 3] = pos
    sp.transform(T)
    return sp
