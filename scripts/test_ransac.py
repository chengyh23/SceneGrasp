import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import sys
from common.utils.nocs_utils import load_depth
from common.utils.misc_utils import (
    convert_realsense_rgb_depth_to_o3d_pcl,
    get_o3d_pcd_from_np,
    get_scene_grasp_model_params,
)
from common.utils.scene_grasp_utils import (
    SceneGraspModel,
    get_final_grasps_from_predictions_np,
    get_grasp_vis,
)
from scene_grasp.scene_grasp_net.utils.nocs_eval_utils import compute_mAP
from scripts.class2synsetId import CLS2SYNSET_ID

CORSAIR_DIR = "/home/chengyh23/Documents/CORSAIR"
sys.path.append(CORSAIR_DIR)
from typing import List
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from tqdm import tqdm
import os
import pandas as pd
from model import fc
from model import load_model
from src.config import get_config
from utils.Info.CADLib import CustomizeCADLib

def get_demo_data_generator(demo_data_path):
    camera_k = np.loadtxt(demo_data_path / "camera_k.txt")

    for color_img_path in demo_data_path.rglob("*_color.png"):
        depth_img_path = color_img_path.parent / (
            color_img_path.stem.split("_")[0] + "_depth.png"
        )
        color_img = cv2.imread(str(color_img_path))  # type:ignore
        depth_img = load_depth(str(depth_img_path))
        yield color_img, depth_img, camera_k
        
def get_model_ids(synset_id: str, split="all") -> List[str]:
    """Get all models' Id in category synset_id, under fullset or train/val/test split subset
    Ref. get_model_ids in ~/CORSAIR/src/corsair/datasets/shapenet/dataset_pcd.py
    """
    split_dir = os.path.join("/home/chengyh23/Documents/CORSAIR/src/corsair/datasets/shapenet", "split")
    _split_df = pd.read_csv(os.path.join(split_dir, f"shapenet_{split}.csv"), dtype=str)
    if synset_id is None:
        return _split_df["modelId"].unique().tolist()
    return _split_df[_split_df["synsetId"] == synset_id]["modelId"].tolist()

def corsair_model(base_data, config, synset_id, model_id):
    """
    Ref. https://github.com/ExistentialRobotics/CORSAIR/blob/main/evaluation.py

    Args:
        base_data (_type_): _description_
        config (_type_): _description_
        synset_id (_type_): TODO At inference, to retrive, how do you now its category??
        model_id (_type_): ground truth modelId
    """
    # feature extraction network for registration
    model = load_model("ResUNetBN2C")(
        in_channels=1,
        out_channels=16,
        bn_momentum=0.05,
        normalize_feature=True,
        conv1_kernel_size=3,
        D=3,
    )
    model = model.to(config.device)
    # embedding network for retrieval
    embedding = fc.conv1_max_embedding(1024, 512, 256).to(config.device)

    # load weights
    checkpoint = torch.load(config.checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])
    embedding.load_state_dict(checkpoint["embedding_state_dict"])
    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    model.eval()
    embedding.eval()
    
    # Extract Retrieval Features for objects
    base_input = ME.SparseTensor(base_data["base_feat"],base_data["base_coords"], device=config.device)
    base_output, base_feat = model(base_input)
    base_feat = embedding(base_feat)
    
    # Extract Retrieval Features for CAD models
    split = "test"  # TODO "all"
    model_ids = get_model_ids(synset_id, split)
    if model_id in model_ids:
        print(f"{model_id} in {synset_id}'s {split} set")
    else:
        print(f"{model_id} not in {synset_id}'s {split} set")
    return
    table_path = "/home/chengyh23/Documents/CORSAIR-ERL/configs/{}_scan2cad.npy".format(synset_id)
    table_path = "/home/chengyh23/Documents/CORSAIR-ERL/configs/{}_scan2cad.npy".format("03001627")
    table_path = "/media/sdb2/chengyh23/k8s/ShapeNetCore.v2.PCD.chamfer_distance/sample15000/ncc/dual_direction/{}/{}.npy".format(synset_id, split)
    cadlib = CustomizeCADLib(config.root, synset_id, model_ids, table_path,
                                      config.voxel_size, preload=config.preload)
    cadlib_loader = torch.utils.data.DataLoader(cadlib, batch_size=config.batch_size, 
                                        shuffle=False, num_workers=4, collate_fn=cadlib.collate_pair_fn)
    lib_feats = []
    lib_outputs = []
    lib_Ts = []
    lib_origins = []
    print("Updating global feature in the CAD library")
    with torch.no_grad():
        for data in tqdm(cadlib_loader, ncols=80):
            base_input = ME.SparseTensor(
                data["base_feat"],
                data["base_coords"],
                device=config.device
            )

            lib_Ts.append(data["base_T"])
            batch_size = len(data["base_idx"])

            base_output, base_feat = model(base_input)
            base_feat = embedding(base_feat)

            for i in range(batch_size):
                base_mask = base_output.C[:, 0] == i
                lib_outputs.append(base_output.F[base_mask, :])
                lib_origins.append(data["base_origin"].to(config.device)[base_mask, :])

            base_feat_norm = nn.functional.normalize(base_feat, dim=1)
            lib_feats.append(base_feat_norm)
    lib_feats = torch.cat(lib_feats, dim=0)
    lib_Ts = torch.cat(lib_Ts, dim=0)
    
    # Evaluate Retrieval Results
    scan2cad_retrieval_eval(base_feat, lib_feats, best_match, cadlib.table, pos_n)
    
    # Registration
    sym_pose
    t_loss, r_loss = eval_pose(T_est)
    
def wrap_pc(rot_coords, config):
    """
    Convert to Minkowski Engine's format
    Ref. CustomizeCADLib::__getitem__
    """
    def quant(rot_coords, config):
        voxel_size = config.voxel_size
        if ME.__version__ >= "0.5.4":
            unique_idx = ME.utils.sparse_quantize(
                np.floor(rot_coords / voxel_size),
                return_index=True,
                return_maps_only=True,
            )
        else:
            unique_idx = ME.utils.sparse_quantize(
                np.floor(rot_coords / voxel_size), return_index=True
            )
        rot_coords = rot_coords[unique_idx, :]
        # coords = coords[unique_idx, :]
        rot_coords_grid = np.floor(rot_coords / voxel_size)

        return rot_coords, rot_coords_grid #, coords
    
    rot_base_coords, rot_base_coords_grid = quant(rot_coords, config)
    base_feat = np.ones([len(rot_base_coords), 1])
    identity = np.eye(4)
    base = {
        "coord": rot_base_coords_grid,
        "origin": rot_base_coords,
        "feat": base_feat,
        "T": identity,
        # "idx": idx,
    }
    
    def collate_pair_fn(list_data):
        
        base_dict = list_data

        base_coords = []
        base_feat = []
        base_T = []
        base_origin = []
        base_idx = []

        for idx in range(len(base_dict)):

            base_coords.append(torch.from_numpy(base_dict[idx]["coord"]))
            base_origin.append(torch.from_numpy(base_dict[idx]["origin"]))
            base_feat.append(torch.from_numpy(base_dict[idx]["feat"]))
            # base_idx.append(base_dict[idx]["idx"])
            base_T.append(torch.from_numpy(base_dict[idx]["T"]))

        batch_base_coords, batch_base_feat = ME.utils.sparse_collate(
            base_coords, base_feat
        )

        data = {}

        data["base_coords"] = batch_base_coords.int()
        data["base_feat"] = batch_base_feat.float()
        data["base_origin"] = torch.cat(base_origin, 0).float()
        # data["base_idx"] = torch.Tensor(base_idx).int()
        data["base_T"] = torch.stack(base_T, 0).float()

        return data
    list_data = [base]
    data = collate_pair_fn(list_data)
    return data

def main(hparams):
    TOP_K = 200  # TODO: use greedy-nms for top-k to get better distributions!
    # Model:
    print("Loading model from checkpoint: ", hparams.checkpoint)
    scene_grasp_model = SceneGraspModel(hparams)

    data_generator = scene_grasp_model.model.val_dataloader()
    assert data_generator.batch_size == 1   # TODO add batch processing
    demo_data_path = Path("outreach/demo_data")
    camera_k = np.loadtxt(demo_data_path / "camera_k.txt")
    for batch in data_generator:
        # image, seg_target, depth_target, pose_targets, detections_gt, scene_name = batch
        image, seg_target, modelIds, depth_target, pose_targets, detections_gt, scene_name = batch
        print("haha")
        pred_dp = scene_grasp_model.get_predictions_from_preprocessed(image, camera_k)
        if pred_dp is None:
            print("No objects found.")
            continue
        # Directly Predict Pose
        T_est =  pred_dp.pose_matrices[0]
        # RANSAC 
        # For all objects detected in a scene,
        # select synsetId==03642806 (laptop)
        # Retrieve from CAD lib
        config = get_config()
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.checkpoint = os.path.join("/home/chengyh23/Documents/CORSAIR-ERL", "ckpts", f"scannet_ret_table_best")
        config.root = "/media/sdb2/chengyh23/ShapeNetCore.v2.PCD.npy/sample15000/ncc"
        assert pred_dp.get_len() == len(modelIds[0])
        for i in range(pred_dp.get_len()):
            # base_pc = pred_dp.endpoints['xyz'][0]
            base_pc = pred_dp.obj_canonical_pcls[i]
            class_id = pred_dp.class_ids[i]
            synset_id = CLS2SYNSET_ID[class_id]
            if synset_id != "03642806":
                continue
            model_id = modelIds[i]
            base_data = wrap_pc(base_pc, config)
            corsair_model(base_data, config, synset_id, model_id)
        # t_loss, r_loss = eval_pose(T_est)
        # compute_mAP(pred_dp)
    return
    # ============================
    demo_data_path = Path("outreach/demo_data")
    data_generator = get_demo_data_generator(demo_data_path)
    for rgb, depth, camera_k in data_generator:
        print("------- Showing results ------------")
        pred_dp = scene_grasp_model.get_predictions(rgb, depth, camera_k)
        if pred_dp is None:
            print("No objects found.")
            continue
        
        all_gripper_vis = []
        for pred_idx in range(pred_dp.get_len()):
            (
                pred_grasp_poses_cam_final,
                pred_grasp_widths,
                _,
            ) = get_final_grasps_from_predictions_np(
                pred_dp.scale_matrices[pred_idx][0, 0],
                pred_dp.endpoints,
                pred_idx,
                pred_dp.pose_matrices[pred_idx],
                TOP_K=TOP_K,
            )

            grasp_colors = np.ones((len(pred_grasp_widths), 3)) * [1, 0, 0]
            all_gripper_vis += [
                get_grasp_vis(
                    pred_grasp_poses_cam_final, pred_grasp_widths, grasp_colors
                )
            ]

        pred_pcls = pred_dp.get_camera_frame_pcls()
        pred_pcls_o3d = []
        for pred_pcl in pred_pcls:
            pred_pcls_o3d.append(get_o3d_pcd_from_np(pred_pcl))
        o3d_pcl = convert_realsense_rgb_depth_to_o3d_pcl(rgb, depth / 1000, camera_k)
        o3d.visualization.webrtc_server.enable_webrtc()
        print(">Showing predicted shapes:")
        o3d.visualization.draw(  # type:ignore
            [o3d_pcl] + pred_pcls_o3d
        )
        print(">Showing predicted grasps:")
        o3d.visualization.draw(  # type:ignore
            pred_pcls_o3d + all_gripper_vis
        )


if __name__ == "__main__":
    args_list = None
    if len(sys.argv) > 1:
        args_list = sys.argv[1:]
    hparams = get_scene_grasp_model_params(args_list)
    main(hparams)
