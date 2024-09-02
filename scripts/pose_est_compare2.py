"""
Aug 2024
Johan Cheng

This script is used to compare pose estimation performance of 
direct prediction VS retrieval/registration-based estimation.
"""
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import sys
import datetime
from common.utils.nocs_utils import load_depth
from common.utils.misc_utils import (
    convert_realsense_rgb_depth_to_o3d_pcl,
    get_o3d_pcd_from_np,
    get_scene_grasp_model_params,
    get_o3d_chamfer_distance,
)
from common.utils.scene_grasp_utils import (
    SceneGraspModel,
    get_final_grasps_from_predictions_np,
    get_grasp_vis,
)
from scene_grasp.scene_grasp_net.utils.matches_utils import get_matches, load_deformnet_nocs_results
from scene_grasp.scene_grasp_net.utils.nocs_eval_utils import compute_mAP
from scripts.class2synsetId import CLS2SYNSET_ID
from scripts.demo_2d import render_predcitions

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
from utils.visualize import Wvisualize  # CORSAIR
from utils.preprocess import apply_transform    # CORSAIR

from scripts.rgbd2pcls import get_object_pc_from_scene
from scripts.pose_est_compare import corsair_model2
from scripts.demo_compare import normalize_pc
def main(hparams, ):
    """
    Get pose estimation from direct prediction and retrieval/registration based,
    compute RTE/RRE of both methods and write to file.
    
    Ref. scripts/demo.py for getting predictions from SceneGrasp
    """
    TOP_K = 200  # TODO: use greedy-nms for top-k to get better distributions!
    # class_of_interest = 4   # can
    class_of_interest = 5   # laptop
    # SceneGrasp Model:
    print("Loading model from checkpoint: ", hparams.checkpoint)
    scene_grasp_model = SceneGraspModel(hparams)
    data_generator = scene_grasp_model.model.val_dataloader()
    assert data_generator.batch_size == 1   # TODO add batch processing
    demo_data_path = Path("outreach/demo_data")
    camera_k = np.loadtxt(demo_data_path / "camera_k.txt")
    
    # CORSAIR Model:
    desired_params = {
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        # "resume": os.path.join("/home/chengyh23/Documents/CORSAIR-ERL", "ckpts", f"scannet_ret_table_best"),
        # "resume": os.path.join("/home/chengyh23/Documents/CORSAIR", "src/ckpts", "cat_ret_conv256max_01_FCGFus_can"),
        "resume": os.path.join("/home/chengyh23/Documents/CORSAIR", "src/ckpts", "cat_ret_conv256max_01_FCGFus_laptop"),
        "root": "/media/sdb2/chengyh23/ShapeNetCore.v2.PCD.npy/sample15000/ncc",
        
        "embedding": "conv1_max_embedding",
        "dim": [1024, 512,  256],
        "model": "ResUNetBN2C",
        "model_n_out": 16,
        "normalize_feature": True,
        "conv1_kernel_size": 3,
        "bn_momentum": 0.05,
        
        "catid": CLS2SYNSET_ID[class_of_interest],
    }
    config = get_config(desired_params)
        
    # from ...CORSAIR.src.test_time.FeatureExtractor import FeatureExtractor
    from src.test_time.FeatureExtractor import FeatureExtractor
    feature_extractor = FeatureExtractor(config)
    # from ...CORSAIR.src.test_time.RetrievalModule import RetrievalModule
    from src.test_time.RetrievalModule import RetrievalModule
    retrieval_module_ = RetrievalModule(config, feature_extractor, "shapenet", update=True)

    stat = {
        "img_name": [],
        "pred_idx": [],
        "gt_idx": [],
        "RTE_pred": [],
        "RRE_pred": [],
        "RTE_regis": [],
        "RRE_regis": [],
        "CD_pred": [],
        "CD_regis": [],
    }
    _i=0
    for batch in tqdm(data_generator):
        _i += 1
        if _i > 3000: 
            break
        
        # image, seg_target, depth_target, pose_targets, detections_gt, scene_name = batch
        image, seg_target, modelIds, instance_ids, depth_target, pose_targets, bboxes_gt, _, img_name, scene_name = batch
        modelIds = modelIds[0]
        instance_ids = instance_ids[0]
        img_name = img_name[0]
        # if img_name != "CAMERA/val/02159/0000": continue    # DEBUG
        # if img_name != "CAMERA/val/00392/0005": continue    # DEBUG
        pred_dp = scene_grasp_model.get_predictions_from_preprocessed(image, camera_k)
        if pred_dp is None:
            print(f"[{img_name}] > No objects found.")
            continue
        print(f"[{img_name}] > ")
        
        # render_predcitions(pred_dp, img_name, camera_k, bboxes_gt)
        # print(img_name)
        # # if img_name == "CAMERA/val/00013/0002":
        # if img_name == "CAMERA/val/00000/0009":
        #     print(len(pred_dp.obj_canonical_pcls))
        #     colors = ["BLUE","RED","BLACK"]
        #     colors = colors[:len(pred_dp.obj_canonical_pcls)]
        #     Wvisualize(pred_dp.obj_canonical_pcls, colors)
        
        # >>>>>> Match prediction with ground truth (object detection) >>>>>>
        # compute_mAP(pred_dp)
        degree_thres_list = list(range(0, 61, 1))
        shift_thres_list = [i / 2 for i in range(21)]
        # iou_thres_list = [i / 100 for i in range(101)]
        iou_thres_list = [i / 100 for i in range(3)]
        nocs = load_deformnet_nocs_results(img_name, "data/deformnet_eval/nocs_results/")
        iou_pred_matches_all, pose_pred_matches_all, iou_gt_matches_all, pose_gt_matches_all = get_matches(
            pred_dp, nocs, "results/",
            degree_thres_list, shift_thres_list, iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=False)
        
        gt_class_ids = nocs['gt_class_ids']
        pred_class_ids = np.array(pred_dp.class_ids)
        print("objects: ", gt_class_ids, pred_class_ids) # DEBUG
        print("matches: ", iou_pred_matches_all[0,:], iou_gt_matches_all[0,:])   # DEBUG
        
        # >>>>>> Compare direct pose prediction VS RANSAC pose estimation >>>>>>
        from scripts.class2synsetId import WORD2CLS_ID, WORD2SYNSET_ID
        # assert pred_dp.get_len() == len(modelIds)
        # for i in range(pred_dp.get_len()):
            # Filter class of interest AND instance matched well with ground truth
        #     class_id = pred_dp.class_ids[i]
        #     synset_id = CLS2SYNSET_ID[class_id]
        #     if synset_id != "03642806":
        #         continue
        #     if iou_pred_matches_all[i] == -1:
        #         continue
        n_ooi = 0   # number of objects of interest
        for pred_idx, gt_idx in enumerate(iou_pred_matches_all[0, :]):
            # Filter class of interest AND instance matched well with ground truth
            if pred_class_ids[pred_idx] != class_of_interest:
                continue
            if gt_idx == -1:
                continue
            print("New OOI! ")
            n_ooi += 1
            # base_pc = pred_dp.obj_canonical_pcls[pred_idx]
            base_pc = get_object_pc_from_scene(img_name, instance_ids[gt_idx])
            
            # 1. Directly Predict Pose
            from utils.eval_pose import eval_pose_my
            T_est =  pred_dp.pose_matrices[pred_idx, :, :]
            T_target = nocs['gt_RTs'][gt_idx, :, :]   # 4,4
            # T_target = nocs['gt_scales'][pred_idx, :]   # 3
            t_loss1, r_loss1 = eval_pose_my(T_est, T_target)
            print('[Direct pose prediction] RTE {0:.2f}, RRE {1:.2f}'.format(t_loss1, r_loss1))
            cd_pred = get_o3d_chamfer_distance(
                get_o3d_pcd_from_np(apply_transform(pred_dp.obj_canonical_pcls[pred_idx], T_est), color=[1,0,1]), 
                get_o3d_pcd_from_np(base_pc, color=[0.2,0.2,0.2])
                )
            
            # 2. Give to CORSAIR
            pc_center, r, offset = normalize_pc(base_pc)
            T_norm = np.eye(4)
            T_norm[0:3, 0:3] = r * np.eye(3)
            T_norm[0:3, 3] = offset
            
            model_id = modelIds[gt_idx]
            synset_id = CLS2SYNSET_ID[class_of_interest]
            # t_loss, r_loss = corsair_model(base_pc, config, synset_id, model_id)
            # t_loss2, r_loss2 = corsair_model2(base_pc, config, synset_id, model_id, feature_extractor, retrieval_module_)
            T_reg, pos_coords = corsair_model2(pc_center, config, synset_id, model_id, feature_extractor, retrieval_module_)
            T_reg = T_reg.numpy()
            T_est = T_norm @ np.linalg.inv(T_reg)
            # T_target: from ShapeNet to scene
            Ta = np.eye(4)
            Ta[2, 2] = -1   # flip z axis
            Tb = np.eye(4)
            Tb[0:3, 0:3] = np.array(
                [[0,0,1], 
                [0,1,0], 
                [1,0,0]]
                )
            T_ShapeNet2PredCanonical = Tb @ Ta
            T_target2 = T_target @ Tb @ Ta
            t_loss2, r_loss2 = eval_pose_my(T_est, T_target2)
            print('[Registration-based pose prediction] RTE {0:.2f}, RRE {1:.2f}'.format(t_loss2, r_loss2))
            cd_retrieved = get_o3d_chamfer_distance(
                get_o3d_pcd_from_np(apply_transform(pos_coords, T_est), color=[1,0,1]), 
                get_o3d_pcd_from_np(base_pc, color=[0.2,0.2,0.2])
                )
            
            stat["img_name"].append(img_name)
            stat["pred_idx"].append(pred_idx)
            stat["gt_idx"].append(gt_idx)
            stat["RTE_pred"].append(t_loss1)
            stat["RRE_pred"].append(r_loss1)
            stat["RTE_regis"].append(t_loss2)
            stat["RRE_regis"].append(r_loss2)
            stat["CD_pred"].append(cd_pred)
            stat["CD_regis"].append(cd_retrieved)
            print(cd_pred, cd_retrieved)

        print("Total # OOI: ", n_ooi)
        
        
    # Save results
    ckpt_name = os.path.split(config.resume)[1]
    catid = CLS2SYNSET_ID[class_of_interest]
    now = datetime.datetime.now()
    time_str = now.strftime('%Y%m%d_%H%M%S')
    out_stat_filename = f"{ckpt_name}-{catid}-{time_str}.csv"
    df = pd.DataFrame(stat)
    df.to_csv(os.path.join("results/pose/", out_stat_filename), index=False)
    print("Write to {}".format(os.path.join("results/pose/", out_stat_filename)))
    return

if __name__ == "__main__":
    # # Write stat
    # args_list = None
    # if len(sys.argv) > 1:
    #     args_list = sys.argv[1:]
    # hparams = get_scene_grasp_model_params(args_list)
    # main(hparams)
    # # objects_stat(hparams)
    
    from scripts.pose_est_compare import organize_result
    # Read stat
    # in_stat_filename = "scannet_ret_table_best-03642806-20240827_222805.csv"
    in_stat_filename = "cat_ret_conv256max_01_FCGFus_laptop-03642806-20240828_115535.csv"
    organize_result(in_stat_filename)