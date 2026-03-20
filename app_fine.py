import gradio as gr
from gradio_litmodel3d import LitModel3D

import os
import shutil
import uuid
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisVGGTTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

from wheels.vggt.vggt.utils.load_fn import load_and_preprocess_images
from wheels.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
import open3d as o3d
from torchvision import transforms as TF
from PIL import Image
import sys
sys.path.append("wheels")
from wheels.mast3r.model import AsymmetricMASt3R
from wheels.mast3r.fast_nn import fast_reciprocal_NNs
from wheels.dust3r.dust3r.inference import inference
from wheels.dust3r.dust3r.utils.image import load_images_new
from trellis.utils.general_utils import *
import copy

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image for 3D generation.
    
    This function is called when a user uploads an image or selects an example.
    It applies background removal and other preprocessing steps necessary for
    optimal 3D model generation.

    Args:
        image (Image.Image): The input image from the user

    Returns:
        Image.Image: The preprocessed image ready for 3D generation
    """
    processed_image = pipeline.preprocess_image(image)
    return processed_image

def preprocess_videos(video: str) -> List[Tuple[Image.Image, str]]:
    """
    Preprocess the input video for multi-image 3D generation.
    
    This function is called when a user uploads a video.
    It extracts frames from the video and processes each frame to prepare them
    for the multi-image 3D generation pipeline.
    
    Args:
        video (str): The path to the input video file
        
    Returns:
        List[Tuple[Image.Image, str]]: The list of preprocessed images ready for 3D generation
    """
    vid = imageio.get_reader(video, 'ffmpeg')
    fps = vid.get_meta_data()['fps']
    images = []
    for i, frame in enumerate(vid):
        if i % max(int(fps * 1), 1) == 0:
            img = Image.fromarray(frame)
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            images.append(img)
    vid.close()
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images

def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    """
    Preprocess a list of input images for multi-image 3D generation.
    
    This function is called when users upload multiple images in the gallery.
    It processes each image to prepare them for the multi-image 3D generation pipeline.
    
    Args:
        images (List[Tuple[Image.Image, str]]): The input images from the gallery
        
    Returns:
        List[Image.Image]: The preprocessed images ready for 3D generation
    """
    images = [image[0] for image in images]
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    
    
def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed for generation.
    
    This function is called by the generate button to determine whether to use
    a random seed or the user-specified seed value.
    
    Args:
        randomize_seed (bool): Whether to generate a random seed
        seed (int): The user-specified seed value
        
    Returns:
        int: The seed to use for generation
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

def align_camera(num_frames, extrinsic, intrinsic, rend_extrinsics, rend_intrinsics):

    extrinsic_tmp = extrinsic.clone()
    camera_relative = torch.matmul(extrinsic_tmp[:num_frames,:3,:3].permute(0,2,1), extrinsic_tmp[num_frames:,:3,:3])
    camera_relative_angle = torch.acos(((camera_relative[:,0,0] + camera_relative[:,1,1] + camera_relative[:,2,2] - 1) / 2).clamp(-1, 1))
    idx = torch.argmin(camera_relative_angle)
    target_extrinsic = rend_extrinsics[idx:idx+1].clone()

    focal_x = intrinsic[:num_frames,0,0].mean()
    focal_y = intrinsic[:num_frames,1,1].mean()
    focal = (focal_x + focal_y) / 2
    rend_focal = (rend_intrinsics[0][0,0] + rend_intrinsics[0][1,1]) * 518 / 2
    focal_scale = rend_focal / focal
    target_intrinsic = intrinsic[num_frames:].clone()
    fxy = (target_intrinsic[:,0,0] + target_intrinsic[:,1,1]) / 2 * focal_scale 
    target_intrinsic[:,0,0] = fxy
    target_intrinsic[:,1,1] = fxy
    return target_extrinsic, target_intrinsic

def refine_pose_mast3r(rend_image_pil, target_image_pil, original_size, fxy, target_extrinsic, rend_depth):
    images_mast3r = load_images_new([rend_image_pil, target_image_pil], size=512, square_ok=True)
    with torch.no_grad():
        output = inference([tuple(images_mast3r)], mast3r_model, device, batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    del output
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
    scale_x = original_size[1] / W0.item()
    scale_y = original_size[0] / H0.item()
    for pixel in matches_im1:
        pixel[0] *= scale_x
        pixel[1] *= scale_y
    for pixel in matches_im0:
        pixel[0] *= scale_x
        pixel[1] *= scale_y
    depth_map = rend_depth[0]
    fx, fy, cx, cy = fxy.item(), fxy.item(), original_size[1]/2, original_size[0]/2  # Example values for focal lengths and principal point
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    dist_eff = np.array([0,0,0,0], dtype=np.float32)
    predict_c2w_ini = np.linalg.inv(target_extrinsic[0].cpu().numpy())
    predict_w2c_ini = target_extrinsic[0].cpu().numpy()
    initial_rvec, _ = cv2.Rodrigues(predict_c2w_ini[:3,:3].astype(np.float32))
    initial_tvec = predict_c2w_ini[:3,3].astype(np.float32)
    K_inv = np.linalg.inv(K)
    height, width = depth_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    depth_flat = depth_map.flatten()
    x_normalized = (x_flat - K[0, 2]) / K[0, 0]
    y_normalized = (y_flat - K[1, 2]) / K[1, 1]
    X_camera = depth_flat * x_normalized
    Y_camera = depth_flat * y_normalized
    Z_camera = depth_flat
    points_camera = np.vstack((X_camera, Y_camera, Z_camera, np.ones_like(X_camera)))
    points_world = predict_c2w_ini @ points_camera
    X_world = points_world[0, :]
    Y_world = points_world[1, :]
    Z_world = points_world[2, :]
    points_3D = np.vstack((X_world, Y_world, Z_world))
    scene_coordinates_gs = points_3D.reshape(3, original_size[0], original_size[1])
    points_3D_at_pixels = np.zeros((matches_im0.shape[0], 3))
    for i, (x, y) in enumerate(matches_im0):
        points_3D_at_pixels[i] = scene_coordinates_gs[:, y, x]

    success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D_at_pixels.astype(np.float32), matches_im1.astype(np.float32), K, \
                                                      dist_eff,rvec=initial_rvec,tvec=initial_tvec, useExtrinsicGuess=True, reprojectionError=1.0,\
                                                      iterationsCount=2000,flags=cv2.SOLVEPNP_EPNP)
    R = perform_rodrigues_transformation(rvec)
    trans = -R.T @ np.matrix(tvec)
    predict_c2w_refine = np.eye(4)
    predict_c2w_refine[:3,:3] = R.T
    predict_c2w_refine[:3,3] = trans.reshape(3)
    target_extrinsic_final = torch.tensor(predict_c2w_refine).inverse().cuda()[None].float()
    return target_extrinsic_final

def pointcloud_registration(rend_image_pil, target_image_pil, original_size, 
                            fxy, target_extrinsic, rend_depth, target_pointmap, 
                            down_pcd, pcd):
    images_mast3r = load_images_new([rend_image_pil, target_image_pil], size=512, square_ok=True)
    with torch.no_grad():
        output = inference([tuple(images_mast3r)], mast3r_model, device, batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    del output
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
    scale_x = original_size[1] / W0.item()
    scale_y = original_size[0] / H0.item()
    for pixel in matches_im1:
        pixel[0] *= scale_x
        pixel[1] *= scale_y
    for pixel in matches_im0:
        pixel[0] *= scale_x
        pixel[1] *= scale_y
    depth_map = rend_depth[0]
    fx, fy, cx, cy = fxy.item(), fxy.item(), original_size[1]/2, original_size[0]/2  # Example values for focal lengths and principal point
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    dist_eff = np.array([0,0,0,0], dtype=np.float32)
    predict_c2w_ini = np.linalg.inv(target_extrinsic[0].cpu().numpy())
    predict_w2c_ini = target_extrinsic[0].cpu().numpy()
    initial_rvec, _ = cv2.Rodrigues(predict_c2w_ini[:3,:3].astype(np.float32))
    initial_tvec = predict_c2w_ini[:3,3].astype(np.float32)
    K_inv = np.linalg.inv(K)
    height, width = depth_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    depth_flat = depth_map.flatten()
    x_normalized = (x_flat - K[0, 2]) / K[0, 0]
    y_normalized = (y_flat - K[1, 2]) / K[1, 1]
    X_camera = depth_flat * x_normalized
    Y_camera = depth_flat * y_normalized
    Z_camera = depth_flat
    points_camera = np.vstack((X_camera, Y_camera, Z_camera, np.ones_like(X_camera)))
    points_world = predict_c2w_ini @ points_camera
    X_world = points_world[0, :]
    Y_world = points_world[1, :]
    Z_world = points_world[2, :]
    points_3D = np.vstack((X_world, Y_world, Z_world))
    scene_coordinates_gs = points_3D.reshape(3, original_size[0], original_size[1])
    points_3D_at_pixels = np.zeros((matches_im0.shape[0], 3))
    for i, (x, y) in enumerate(matches_im0):
        points_3D_at_pixels[i] = scene_coordinates_gs[:, y, x]
    
    points_3D_at_pixels_2 = np.zeros((matches_im1.shape[0], 3))
    for i, (x, y) in enumerate(matches_im1):
        points_3D_at_pixels_2[i] = target_pointmap[:, y, x]

    dist_1 = np.linalg.norm(points_3D_at_pixels - points_3D_at_pixels.mean(axis=0), axis=1)
    scale_1 = dist_1[dist_1 < np.percentile(dist_1, 99)].mean()
    dist_2 = np.linalg.norm(points_3D_at_pixels_2 - points_3D_at_pixels_2.mean(axis=0), axis=1)
    scale_2 = dist_2[dist_2 < np.percentile(dist_2, 99)].mean()
    # scale_1 = np.linalg.norm(points_3D_at_pixels - points_3D_at_pixels.mean(axis=0), axis=1).mean()
    # scale_2 = np.linalg.norm(points_3D_at_pixels_2 - points_3D_at_pixels_2.mean(axis=0), axis=1).mean()
    points_3D_at_pixels_2 = points_3D_at_pixels_2 * (scale_1 / scale_2)
    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(points_3D_at_pixels)
    pcd_2 = o3d.geometry.PointCloud()
    pcd_2.points = o3d.utility.Vector3dVector(points_3D_at_pixels_2)
    indices = np.arange(points_3D_at_pixels.shape[0])
    correspondences = np.stack([indices, indices], axis=1)
    correspondences = o3d.utility.Vector2iVector(correspondences)
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd_2,
        pcd_1,
        correspondences,
        0.03,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=5,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 10000),
    )
    transformation_matrix = result.transformation.copy()
    transformation_matrix[:3,:3] = transformation_matrix[:3,:3] * (scale_1 / scale_2)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        down_pcd, pcd, 0.02, transformation_matrix
    )
    return transformation_matrix, evaluation.fitness

def generate_and_extract_glb(
    multiimages: List[Tuple[Image.Image, str]],
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    ss_guidance_rescale: float,
    ss_rescale_t: float,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    slat_guidance_rescale: float,
    slat_rescale_t: float,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    mesh_simplify: float,
    texture_size: int,
    refine: Literal["Yes", "No"],
    ss_refine: Literal["noise", "deltav", "No"],
    registration_num_frames: int,
    trellis_stage1_lr: float, 
    trellis_stage1_start_t: float,  
    trellis_stage2_lr: float, 
    trellis_stage2_start_t: float,
    req: gr.Request,
) -> Tuple[dict, str, str, str]:
    """
    Convert an image to a 3D model and extract GLB file.

    Args:
        image (Image.Image): The input image.
        multiimages (List[Tuple[Image.Image, str]]): The input images in multi-image mode.
        is_multiimage (bool): Whether is in multi-image mode.
        seed (int): The random seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.
        multiimage_algo (Literal["multidiffusion", "stochastic"]): The algorithm for multi-image generation.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
        str: The path to the extracted GLB file.
        str: The path to the extracted GLB file (for download).
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    image_files = [image[0] for image in multiimages]

    # Generate 3D model
    outputs, coords, ss_noise = pipeline.run(
        image=image_files,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": slat_guidance_rescale,
            "rescale_t": slat_rescale_t,
        },
        mode=multiimage_algo,
    )
    if refine == "Yes":
        try:
            images, alphas = load_and_preprocess_images(multiimages)
            images, alphas = images.to(device), alphas.to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=pipeline.VGGT_dtype):
                    images = images[None]
                    aggregated_tokens_list, ps_idx = pipeline.VGGT_model.aggregator(images)
                # Predict Cameras
                pose_enc = pipeline.VGGT_model.camera_head(aggregated_tokens_list)[-1]
                # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                # Predict Point Cloud
                point_map, point_conf = pipeline.VGGT_model.point_head(aggregated_tokens_list, images, ps_idx)
                del aggregated_tokens_list
                mask = (alphas[:,0,...][...,None] > 0.8)
                conf_threshold = np.percentile(point_conf.cpu().numpy(), 50)
                confidence_mask = (point_conf[0] > conf_threshold) & (point_conf[0] > 1e-5)
                mask = mask & confidence_mask[...,None]
                point_map_by_unprojection = point_map[0]
                point_map_clean = point_map_by_unprojection[mask[...,0]]
                center_point = point_map_clean.mean(0)
                scale = np.percentile((point_map_clean - center_point[None]).norm(dim=-1).cpu().numpy(), 98)
                outlier_mask = (point_map_by_unprojection - center_point[None]).norm(dim=-1) <= scale
                final_mask = mask & outlier_mask[...,None]
                point_map_perframe = (point_map_by_unprojection - center_point[None, None, None]) / (2 * scale)
                point_map_perframe[~final_mask[...,0]] = 127/255
                point_map_perframe = point_map_perframe.permute(0,3,1,2)
                images = images[0].permute(0,2,3,1)
                images[~(alphas[:,0,...][...,None] > 0.8)[...,0]] = 0.
                input_images = images.permute(0,3,1,2).clone()
                vggt_extrinsic = extrinsic[0]
                vggt_extrinsic = torch.cat([vggt_extrinsic, torch.tensor([[[0,0,0,1]]]).repeat(vggt_extrinsic.shape[0], 1, 1).to(vggt_extrinsic)], dim=1)
                vggt_intrinsic = intrinsic[0]
                vggt_intrinsic[:,:2] = vggt_intrinsic[:,:2] / 518
                vggt_extrinsic[:,:3,3] = (torch.matmul(vggt_extrinsic[:,:3,:3], center_point[None,:,None].float())[...,0] + vggt_extrinsic[:,:3,3]) / (2 * scale)
                pointcloud = point_map_perframe.permute(0,2,3,1)[final_mask[...,0]]
                idxs = torch.randperm(pointcloud.shape[0])[:min(50000, pointcloud.shape[0])]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pointcloud[idxs].cpu().numpy())
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=3.0)
                inlier_cloud = pcd.select_by_index(ind)
                outlier_cloud = pcd.select_by_index(ind, invert=True)
                distance = np.array(inlier_cloud.points) - np.array(inlier_cloud.points).mean(axis=0)[None]
                scale = np.percentile(np.linalg.norm(distance, axis=1), 97)
                voxel_size = 1/64*scale*2
                down_pcd = inlier_cloud.voxel_down_sample(voxel_size)
            torch.cuda.empty_cache()

            video, rend_extrinsics, rend_intrinsics = render_utils.render_multiview(outputs['gaussian'][0], num_frames=registration_num_frames)
            rend_extrinsics = torch.stack(rend_extrinsics, dim=0)
            rend_intrinsics = torch.stack(rend_intrinsics, dim=0)
            target_extrinsics = []
            target_intrinsics = []
            target_transforms = []
            target_fitnesses = []   
            pcd = o3d.geometry.PointCloud()
            mesh = outputs['mesh'][0]
            idxs = torch.randperm(mesh.vertices.shape[0])[:min(50000, mesh.vertices.shape[0])]
            pcd.points = o3d.utility.Vector3dVector(mesh.vertices[idxs].cpu().numpy())
            distance = np.array(pcd.points) - np.array(pcd.points).mean(axis=0)[None]
            scale = np.linalg.norm(distance, axis=1).max()
            voxel_size = 1/64*scale*2
            pcd = pcd.voxel_down_sample(voxel_size)
            # pcd.points = o3d.utility.Vector3dVector((coords[:,1:].cpu().numpy() + 0.5) / 64 - 0.5)

            for k in range(len(image_files)):
                images = torch.stack([TF.ToTensor()(render_image) for render_image in video['color']] + [TF.ToTensor()(image_files[k].convert("RGB"))], dim=0)
                # if len(images) == 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=pipeline.VGGT_dtype):
                        # predictions = vggt_model(images.cuda())
                        aggregated_tokens_list, ps_idx = pipeline.VGGT_model.aggregator(images[None].cuda())
                    pose_enc = pipeline.VGGT_model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                extrinsic, intrinsic = extrinsic[0], intrinsic[0]
                extrinsic = torch.cat([extrinsic, torch.tensor([0,0,0,1])[None,None].repeat(extrinsic.shape[0], 1, 1).to(extrinsic.device)], dim=1)
                del aggregated_tokens_list, ps_idx

                target_extrinsic, target_intrinsic = align_camera(registration_num_frames, extrinsic, intrinsic, rend_extrinsics, rend_intrinsics)
                fxy = target_intrinsic[:,0,0]
                target_intrinsic_tmp = target_intrinsic.clone()
                target_intrinsic_tmp[:,:2] = target_intrinsic_tmp[:,:2] / 518

                target_extrinsic_list = [target_extrinsic]
                iou_list = []
                iterations = 3
                for i in range(iterations + 1):
                    j = 0
                    rend = render_utils.render_frames(outputs['gaussian'][0], target_extrinsic, target_intrinsic_tmp, {'resolution': 518, 'bg_color': (0, 0, 0)}, need_depth=True)
                    rend_image = rend['color'][j] # (518, 518, 3)
                    rend_depth = rend['depth'][j] # (3, 518, 518)

                    depth_single = rend_depth[0].astype(np.float32)   # (H, W)
                    mask = (depth_single != 0).astype(np.uint8)  # 
                    kernel = np.ones((3, 3), np.uint8)
                    mask_eroded = cv2.erode(mask, kernel, iterations=3)
                    depth_eroded = depth_single * mask_eroded
                    rend_depth_eroded = np.stack([depth_eroded]*3, axis=0)

                    rend_image = torch.tensor(rend_image).permute(2,0,1) / 255
                    target_image = images[registration_num_frames:].to(target_extrinsic.device)[j]
                    original_size = (rend_image.shape[1], rend_image.shape[2])
                    
                    # import torchvision
                    # torchvision.utils.save_image(rend_image, 'rend_image_{}.png'.format(k))
                    # torchvision.utils.save_image(target_image, 'target_image_{}.png'.format(k))
                    
                    mask_rend = (rend_image.detach().cpu() > 0).any(dim=0)
                    mask_target = (target_image.detach().cpu() > 0).any(dim=0)
                    intersection = (mask_rend & mask_target).sum().item()
                    union = (mask_rend | mask_target).sum().item()
                    iou = intersection / union if union > 0 else 0.0
                    iou_list.append(iou)

                    if i == iterations:
                        break

                    rend_image = rend_image * torch.from_numpy(mask_eroded[None]).to(rend_image.device)
                    rend_image_pil = Image.fromarray((rend_image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                    target_image_pil = Image.fromarray((target_image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                    target_extrinsic[j:j+1] = refine_pose_mast3r(rend_image_pil, target_image_pil, original_size, fxy[j:j+1], target_extrinsic[j:j+1], rend_depth_eroded)    
                    target_extrinsic_list.append(target_extrinsic[j:j+1])
                
                idx = iou_list.index(max(iou_list))
                target_extrinsic[j:j+1] = target_extrinsic_list[idx]
                target_transform, fitness = pointcloud_registration(rend_image_pil, target_image_pil, original_size, fxy[j:j+1], target_extrinsic[j:j+1], \
                                                                    rend_depth_eroded, point_map_perframe[k].cpu().numpy(), down_pcd, pcd)
                target_transforms.append(target_transform)
                target_fitnesses.append(fitness)
                
                target_extrinsics.append(target_extrinsic[j:j+1])
                target_intrinsics.append(target_intrinsic_tmp[j:j+1])
            target_extrinsics = torch.cat(target_extrinsics, dim=0)
            target_intrinsics = torch.cat(target_intrinsics, dim=0)
            
            target_fitnesses_filtered = [x for x in target_fitnesses if x <= 1]
            idx = target_fitnesses.index(max(target_fitnesses_filtered))
            target_transform = target_transforms[idx]
            down_pcd_align = copy.deepcopy(down_pcd).transform(target_transform)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(coords[:,1:].cpu().numpy() / 64 - 0.5)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                down_pcd_align, pcd, 0.02, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 10000))
            down_pcd_align_2 = copy.deepcopy(down_pcd_align).transform(reg_p2p.transformation)
            input_points = torch.tensor(np.asarray(down_pcd_align_2.points)).to(extrinsic.device).float()
            input_points = ((input_points + 0.5).clip(0, 1) * 64 - 0.5).to(torch.int32)
            
            outputs = pipeline.run_refine(
                image=image_files,
                ss_learning_rate=trellis_stage1_lr,
                ss_start_t=trellis_stage1_start_t,
                apperance_learning_rate=trellis_stage2_lr,
                apperance_start_t=trellis_stage2_start_t,
                extrinsics=target_extrinsics,
                intrinsics=target_intrinsics,
                ss_noise=ss_noise,
                input_points=input_points,
                ss_refine_type = ss_refine,
                coords=coords if ss_refine == "No" else None,
                seed=seed,
                formats=["mesh", "gaussian"],
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                    "cfg_interval": [0.6, 1.0],
                    "guidance_rescale": ss_guidance_rescale,
                    "rescale_t": ss_rescale_t,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                    "cfg_interval": [0.6, 1.0],
                    "guidance_rescale": slat_guidance_rescale,
                    "rescale_t": slat_rescale_t,
                },
                mode=multiimage_algo,
            )
        except Exception as e:
            print(f"Error during refinement: {e}")
    # Render video
    video_color = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video_color[i], video_geo[i]], axis=1) for i in range(len(video_color))]
    del video_color, video_geo
    output_id = str(uuid.uuid4())
    video_path = os.path.join(user_dir, f'{output_id}.mp4')
    imageio.mimsave(video_path, video, fps=15)
    del video

    # Extract GLB
    gs = outputs['gaussian'][0]
    mesh = outputs['mesh'][0]
    torch.cuda.empty_cache()
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, f'{output_id}.glb')
    glb.export(glb_path)
    del glb

    # Pack state for optional Gaussian extraction
    state = pack_state(gs, mesh)
    del outputs

    torch.cuda.empty_cache()
    return state, video_path, glb_path, glb_path

@torch.inference_mode()
def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Extract a Gaussian splatting file from the generated 3D model.
    
    This function is called when the user clicks "Extract Gaussian" button.
    It converts the 3D model state into a .ply file format containing
    Gaussian splatting data for advanced 3D applications.

    Args:
        state (dict): The state of the generated 3D model containing Gaussian data
        req (gr.Request): Gradio request object for session management

    Returns:
        Tuple[str, str]: Paths to the extracted Gaussian file (for display and download)
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, f'{uuid.uuid4()}.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


def prepare_multi_example() -> List[Image.Image]:
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        _images = []
        for i in range(1, 9):
            if os.path.exists(f'assets/example_multi_image/{case}_{i}.png'):
                img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
                W, H = img.size
                img = img.resize((int(W / H * 512), 512))
                _images.append(np.array(img))
        if len(_images) > 0:
            images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images


def split_image(image: Image.Image) -> List[Image.Image]:
    """
    Split a multi-view image into separate view images.
    
    This function is called when users select multi-image examples that contain
    multiple views in a single concatenated image. It automatically splits them
    based on alpha channel boundaries and preprocesses each view.
    
    Args:
        image (Image.Image): A concatenated image containing multiple views
        
    Returns:
        List[Image.Image]: List of individual preprocessed view images
    """
    image = np.array(image)
    alpha = image[..., 3]
    alpha = np.any(alpha>0, axis=0)
    start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
    end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image[:, s:e+1]))
    return [preprocess_image(image) for image in images]

# Create interface
demo = gr.Blocks(
    title="ReconViaGen",
    css="""
        .slider .inner { width: 5px; background: #FFF; }
        .viewport { aspect-ratio: 4/3; }
        .tabs button.selected { font-size: 20px !important; color: crimson !important; }
        h1, h2, h3 { text-align: center; display: block; }
        .md_feedback li { margin-bottom: 0px !important; }
    """
)
with demo:
    gr.Markdown("""
    # 💻 ReconViaGen
    <p align="center">
    <a title="Github" href="https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/GAP-LAB-CUHK-SZ/ReconViaGen?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Website" href="https://jiahao620.github.io/reconviagen/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
    </a>
    <a title="arXiv" href="https://jiahao620.github.io/reconviagen/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
    </a>
    </p>

    ✨This demo is partial. We will release the whole model later. Stay tuned!✨
    """)

    with gr.Row():
        with gr.Column():
            with gr.Tabs() as input_tabs:
                with gr.Tab(label="Input Video or Images", id=0) as multiimage_input_tab:
                    input_video = gr.Video(label="Upload Video", interactive=True, height=300)
                    image_prompt = gr.Image(label="Image Prompt", format="png", visible=False, image_mode="RGBA", type="pil", height=300)
                    multiimage_prompt = gr.Gallery(label="Image Prompt", format="png", type="pil", height=300, columns=3)
                    gr.Markdown("""
                        Input different views of the object in separate images.
                    """)

            with gr.Accordion(label="Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=30, step=1)
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                    slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                multiimage_algo = gr.Radio(["stochastic", "multidiffusion"], label="Multi-image Algorithm", value="multidiffusion")
                refine = gr.Radio(["Yes", "No"], label="Refinement of Not", value="Yes")
                ss_refine = gr.Radio(["noise", "deltav", "No"], label="Sparse Structure refinement of not", value="No")
                registration_num_frames = gr.Slider(20, 50, label="Number of frames in registration", value=30, step=1)
                trellis_stage1_lr = gr.Slider(1e-4, 1., label="trellis_stage1_lr", value=1e-1, step=5e-4)
                trellis_stage1_start_t = gr.Slider(0., 1., label="trellis_stage1_start_t", value=0.5, step=0.01)
                trellis_stage2_lr = gr.Slider(1e-4, 1., label="trellis_stage2_lr", value=1e-1, step=5e-4)
                trellis_stage2_start_t = gr.Slider(0., 1., label="trellis_stage2_start_t", value=0.5, step=0.01)

            with gr.Accordion(label="GLB Extraction Settings", open=False):
                mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)

            generate_btn = gr.Button("Generate & Extract GLB", variant="primary")
            extract_gs_btn = gr.Button("Extract Gaussian", interactive=False)
            gr.Markdown("""
                        *NOTE: Gaussian file can be very large (~50MB), it will take a while to display and download.*
                        """)

        with gr.Column():
            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
            model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=300)

            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False)

    output_buf = gr.State()

    # Example images at the bottom of the page
    with gr.Row() as multiimage_example:
        examples_multi = gr.Examples(
            examples=prepare_multi_example(),
            inputs=[image_prompt],
            fn=split_image,
            outputs=[multiimage_prompt],
            run_on_click=True,
            examples_per_page=8,
        )

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    input_video.upload(
        preprocess_videos,
        inputs=[input_video],
        outputs=[multiimage_prompt],
    )
    input_video.clear(
        lambda: tuple([None, None]),
        outputs=[input_video, multiimage_prompt],
    )
    multiimage_prompt.upload(
        preprocess_images,
        inputs=[multiimage_prompt],
        outputs=[multiimage_prompt],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        generate_and_extract_glb,
        inputs=[multiimage_prompt, seed, ss_guidance_strength, ss_sampling_steps, ss_guidance_rescale, ss_rescale_t,
                slat_guidance_strength, slat_sampling_steps, slat_guidance_rescale, slat_rescale_t, multiimage_algo, 
                mesh_simplify, texture_size, refine, ss_refine, registration_num_frames, trellis_stage1_lr, 
                trellis_stage1_start_t, trellis_stage2_lr, trellis_stage2_start_t],
        outputs=[output_buf, video_output, model_output, download_glb],
    ).then(
        lambda: (gr.update(interactive=True), gr.update(interactive=True)),
        outputs=[extract_gs_btn, download_glb],
    )

    video_output.clear(
        lambda: (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)),
        outputs=[extract_gs_btn, download_glb, download_gs],
    )

    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.update(interactive=True),
        outputs=[download_gs],
    )

    model_output.clear(
        lambda: (gr.update(interactive=False), gr.update(interactive=False)),
        outputs=[download_glb, download_gs],
    )
    

# Launch the Gradio app
if __name__ == "__main__":
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained("Stable-X/trellis-vggt-v0-2")
    pipeline.cuda()
    pipeline.VGGT_model.cuda()
    pipeline.birefnet_model.cuda()
    pipeline.dreamsim_model.cuda()
    mast3r_model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").cuda().eval()  
    demo.launch()
