""" 
Johan C.
Aug 25 2024

Convert NOCS RGB-D to pointclouds
"""

import numpy as np
import os
import open3d as o3d
import cv2
from pathlib import Path
from common.utils.nocs_utils import load_depth
import sys
CORSAIR_DIR = "/home/chengyh23/Documents/CORSAIR"
sys.path.append(CORSAIR_DIR)
from utils.visualize import Wvisualize  #CORSAIR

def get_intrinsic():
    demo_data_path = Path("outreach/demo_data")
    camera_k = np.loadtxt(demo_data_path / "camera_k.txt")
    fx, fy, cx, cy = camera_k[0][0], camera_k[1][1], camera_k[0][2], camera_k[1][2]
    return fx, fy, cx, cy

def rgbd2pcls_scene(img_path):
    """Create a pointcloud for a NOCS scene from an RGB-D image and a camera.
    """
    all_exist = os.path.exists(img_path + '_color.png') and \
                os.path.exists(img_path + '_depth.png')
    if not all_exist:
        print("not all exist")
        raise
    img = cv2.imread(img_path + '_color.png')
    depth = load_depth(img_path + '_depth.png')
    
    image = np.array(img.copy())
    img = o3d.geometry.Image(image.astype(np.uint8))
    # depth = np.asarray(depth).astype(np.float32) / cam_scale
    depth = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)
    H, W, _ = image.shape
    fx, fy, cx, cy = get_intrinsic()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    o3d.visualization.webrtc_server.enable_webrtc()
    o3d.visualization.draw(pcd)
    # Wvisualize([pcd],["GREEN"])

def get_object_pc_from_scene(img_name, inst_id):
    root = "data/NOCSDataset/"
    img_path = os.path.join(root, img_name)
    
    all_exist = os.path.exists(img_path + '_color.png') and \
                os.path.exists(img_path + '_depth.png')
    if not all_exist:
        print("not all exist")
        raise
    img = cv2.imread(img_path + '_color.png')
    depth = load_depth(img_path + '_depth.png')
    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[-1] == 255
    del all_inst_ids[-1]    # remove background
    assert inst_id in all_inst_ids
    
    mask_inst = (mask==inst_id)
    depth_inst = mask_inst * depth
    
    image = np.array(img.copy())
    img = o3d.geometry.Image(image.astype(np.uint8))
    # depth = np.asarray(depth).astype(np.float32) / cam_scale
    depth_inst = o3d.geometry.Image(depth_inst)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth_inst)
    H, W, _ = image.shape
    fx, fy, cx, cy = get_intrinsic()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return np.asarray(pcd.points)

def rgbd2pcls_object(img_path):
    """Create pointclouds for all instances in a NOCS scene from an RGB-D image and a camera.
    """
    all_exist = os.path.exists(img_path + '_color.png') and \
                os.path.exists(img_path + '_depth.png')
    if not all_exist:
        print("not all exist")
        raise
    img = cv2.imread(img_path + '_color.png')
    depth = load_depth(img_path + '_depth.png')
    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[-1] == 255
    del all_inst_ids[-1]    # remove background
    num_all_inst = len(all_inst_ids)
    for i in range(num_all_inst):
        if i!=1:
            continue
        mask_inst = mask==all_inst_ids[i]
        depth_inst = mask_inst * depth
        
        image = np.array(img.copy())
        img = o3d.geometry.Image(image.astype(np.uint8))
        # depth = np.asarray(depth).astype(np.float32) / cam_scale
        depth_inst = o3d.geometry.Image(depth_inst)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth_inst)
        H, W, _ = image.shape
        fx, fy, cx, cy = get_intrinsic()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        o3d.visualization.webrtc_server.enable_webrtc()
        o3d.visualization.draw(pcd)
        # Wvisualize([pcd],["GREEN"])
        
def NOCS2ShapeNet(pc_in):
    """from NOCS to ShapeNet coordinate system
    
    Args:
        pc_in (Nx3 numpy.ndarray): [0,1]^3, left-handed coordinate

    Returns:
        pc_out: 0-centered, right-handed coordinate
    """
    pc_out = np.copy(pc_in)
    pc_out[:, 2] = 1 - pc_out[:, 2]
    pc_out -= [0.5, 0.5, 0.5]
    # # # NOCS map
    # # cd1 = coords[:, :, i, :]
    # # # cd1 = cd1[:, :, (2,1,0)]
    # cd1[:, :, 2] = 1 - cd1[:, :, 2]
    # cd1 -= [0.5,0.5,0.5]
    # cd1_pc = cd1[np.any(cd1!= [0, 0, 0], axis=2)]
    # # cd1_pc -= cd1_pc.mean(0)
    # # cd1_pc = cd1_pc / np.max(np.linalg.norm(cd1_pc, 2, 1))
    return pc_out
def ShapeNet2PredCanonical(pc_in):
    """_summary_

    Args:
        pc_in: 0-centered, right-handed coordinate

    Returns:
        pc_out: 0-centered, left-handed coordinate
    """
    pc_out = np.copy(pc_in)
    pc_out[:, 2] = - pc_out[:, 2]
    pc_out = pc_out[:, (2,1,0)]
    
    T1 = np.eye(4)
    T1[2, 2] = -1   # flip z axis
    T2 = np.eye(4)
    T2[0:3, 0:3] = np.array(
        [[0,0,1], 
         [0,1,0], 
         [1,0,0]]
        )
    return pc_out
    
def align_NOCS2ShapeNet(img_path):
    """Align NOCS map pointclouds to ShapeNet CAD model pointclouds

    read NOCS map: Ref. process_data in scene_grasp/scene_grasp_net/data_generation/generate_data_nocs.py
    """

    from scene_grasp.scene_grasp_net.data_generation.generate_data_nocs import process_data
    img_full_path = img_path
    depth_full_path_real = img_full_path + '_depth.png'
    all_exist = os.path.exists(img_full_path + '_color.png') and \
                os.path.exists(img_full_path + '_coord.png') and \
                os.path.exists(img_full_path + '_depth.png') and \
                os.path.exists(img_full_path + '_mask.png') and \
                os.path.exists(img_full_path + '_meta.txt') and \
                os.path.exists(depth_full_path_real)
    if not all_exist:
        print("not all exist")
        raise
    depth = load_depth(depth_full_path_real)
    masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
    
    from utils.preprocess import path_dict  # CORSAIR
    id2path = path_dict("/media/sdb2/chengyh23/ShapeNetCore.v2.PCD.npy/sample15000/ncc/")
    pcs = []
    colors = []
    coord_path = img_full_path + '_coord.png'
    coord_map = cv2.imread(coord_path)[:, :, :3]
    # coord_map = coord_map[:, :, (2, 1, 0)]
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    
    for i in range(len(instance_ids)):
        instance_id = instance_ids[i]
        class_id = class_ids[i]
        model_id = model_list[i]
        print(f"Instance {instance_id}, Class {class_id}, Model {model_id}")
        # DEBUG: NOCS map
        mask_inst = masks[:,:,i]
        mask_inst = np.expand_dims(mask_inst, axis=2)
        cd1 = mask_inst * coord_map
        
        cd1 = cd1[np.any(cd1!= [0, 0, 0], axis=2)]  # TODO [1,1,1]
        cd1_pc = NOCS2ShapeNet(cd1)
        print("NOCS", cd1_pc.mean(0))
        # ShapeNet CAD
        model_path = id2path[model_id]
        pc0 = np.load(model_path)
        print("ShapeNet", pc0.mean(0))
        # pc0 -= pc0.mean(0)
        # pc0 = pc0 / np.max(np.linalg.norm(pc0, 2, 1))
        
        cd1_pc += [1*i,0,0]
        pc0 += [1*i,0,0]
        pcs += [cd1_pc, pc0]
        colors += ["BLUE", "RED"]
        # break
    Wvisualize(pcs, colors)
    # print(coords.shape)

if __name__ == "__main__":
    root = "data/NOCSDataset/CAMERA/"
    img_path = "val/00000/0002"
    img_full_path = os.path.join(root, img_path)
    # rgbd2pcls_object(img_full_path)
    align_NOCS2ShapeNet(img_full_path)