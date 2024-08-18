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
# from ...CORSAIR.utils.visualize import Wvisualize
from utils.visualize import Wvisualize

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

def corsair_model2(base_pc, config, synset_id, model_id, feat_extractor, retrieval_module_):
    # # from ...CORSAIR.src.scene_level import align_scene
    # from src.scene_level import align_object
    # T_est, pcs_, colors_ = align_object(base_pc, feature_extractor, retrieval_module_, pos_sym=1)
    
    base_local_feat, base_global_feat, base_coords = feat_extractor.process(base_pc)
    T0 = np.eye(4)
    
    # Retrieve from library using global feature
    _, topn_idx = retrieval_module_.Top1_my(base_global_feat.detach().cpu().numpy())
    pos_local_feat = retrieval_module_.local_feat_lib[topn_idx[0]]
    pose_coords = retrieval_module_.cadlib[topn_idx[0]]["origin"]
    pose_coord_grid = retrieval_module_.cadlib[topn_idx[0]]["coord"]
    T1 = retrieval_module_.cadlib[topn_idx[0]]["T"]
    # T1 = retrieval_module_.Ts_lib[topn_idx[0], :, :]
    
    # Wvisualize([base_coords, pos_coords], ["BLUE","RED"])  # DEBUG
    # return -1
    
    # Registration using local feature
    # from ...CORSAIR.src.scene_level import sym_pose
    from src.scene_level import sym_pose
    # from ...CORSAIR.utils.eval_pose import eval_pose
    from utils.eval_pose import eval_pose
    T_est = sym_pose(base_local_feat, base_coords, pos_local_feat, pose_coords, pos_sym=2)
    # sym_pose(baseF, xyz0, posF, xyz1, pos_sym, k_nn=5, max_corr=0.20):
    t_loss, r_loss = eval_pose(T_est, T0, T1)
    # print(t_loss, r_loss)
    return t_loss, r_loss
    
def corsair_model(base_pc, config, synset_id, model_id):
    """
    Ref. https://github.com/ExistentialRobotics/CORSAIR/blob/main/evaluation.py
    
    Steps:
    1. Instantiate global/local feature extraction network
    2. Extract Retrieval Features for query
    3. Extract Retrieval Features for CAD models
    4. Retrieve and registration
    
    Args:
        base_data (_type_): _description_
        config (_type_): _description_
        synset_id (_type_): TODO At inference, to retrive, how do you now its category??
        model_id (_type_): ground truth modelId
    Return: 
        t_loss, r_loss
    """
    # Instantiate feature extraction network for registration
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

    # >>>>>>>>>>>>>> Extract Retrieval Features for query >>>>>>>>>>>>>>>
    # from ...CORSAIR.utils.Info.CADLib import SceneGraspCADLib
    from utils.Info.CADLib import SceneGraspCADLib
    cadlib = SceneGraspCADLib(base_pc, config.voxel_size)
    cadlib_loader = torch.utils.data.DataLoader(cadlib, batch_size=config.batch_size, 
                                            shuffle=False, num_workers=4, collate_fn=cadlib.collate_pair_fn)
    base_data = cadlib.__getitem__(0)
    list_data = [base_data]
    base_data = cadlib.collate_pair_fn(list_data)
    
    # Extract Retrieval Features for objects
    base_input = ME.SparseTensor(base_data["base_feat"],base_data["base_coords"], device=config.device)
    base_output, base_feat = model(base_input)
    base_feat = embedding(base_feat)
    base_feat = base_feat.detach().cpu().numpy()
    base_T = base_data["base_T"]
    
    # base_local_feat, base_global_feat, base_coords = feat_extractor.process(base_pc)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # >>>>>>>>>>>>>> Extract Retrieval Features for CAD models >>>>>>>>>>>>>>>
    split = "test"  # TODO "all"
    model_ids = get_model_ids(synset_id, split)
    if model_id in model_ids:
        print(f"{model_id} in {synset_id}'s {split} set")
    else:
        print(f"{model_id} not in {synset_id}'s {split} set")
    # return
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
            tmp_input = ME.SparseTensor(
                data["base_feat"],
                data["base_coords"],
                device=config.device
            )

            lib_Ts.append(data["base_T"])
            batch_size = len(data["base_idx"])

            tmp_output, tmp_feat = model(tmp_input)
            tmp_feat = embedding(tmp_feat)

            for i in range(batch_size):
                tmp_mask = tmp_output.C[:, 0] == i
                lib_outputs.append(tmp_output.F[tmp_mask, :])
                lib_origins.append(data["base_origin"].to(config.device)[tmp_mask, :])

            tmp_feat_norm = nn.functional.normalize(tmp_feat, dim=1)
            lib_feats.append(tmp_feat_norm)
    lib_feats = torch.cat(lib_feats, dim=0)
    lib_Ts = torch.cat(lib_Ts, dim=0)
    
    lib_feats = lib_feats.detach().cpu().numpy()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # Retrieve from library
    # Ref. scan2cad_retrieval_eval
    # or Ref. CORSAIR/src/scene_level.py
    # # Evaluate Retrieval Results
    # scan2cad_retrieval_eval(base_feat, lib_feats, best_match, cadlib.table, pos_n)
    dists = np.linalg.norm(
        base_feat[:, None, :] - lib_feats[None, :, :], ord=2, axis=2
    )
    rank = np.argsort(dists, 1)
    topn_idx = rank[0,:5]

    # similar_pcs = [self.pc_names[i] for i in topn_idx]
        
    # Registration
    # from ...CORSAIR.src.scene_level import sym_pose
    from src.scene_level import sym_pose
    # from ...CORSAIR.utils.eval_pose import eval_pose
    from utils.eval_pose import eval_pose
    # xyz0 = 
    T0 = base_T[0, :, :]
    xyz1 = cadlib[topn_idx[0]]["origin"]
    xyz1_coord_grid = cadlib[topn_idx[0]]["coord"]
    T1 = lib_Ts[topn_idx[0], :, :]
    
    # Wvisualize([base_data["base_origin"], xyz1], ["BLUE","RED"])  # DEBUG
    # # Wvisualize([base_coords_grid, xyz1_coord_grid], ["BLUE","RED"])
    # return -1
    
    T_est = sym_pose(base_output.F, base_data["base_origin"], lib_outputs[topn_idx[0]], xyz1, pos_sym=2)
    # sym_pose(baseF, xyz0, posF, xyz1, pos_sym, k_nn=5, max_corr=0.20):
    t_loss, r_loss = eval_pose(T_est, T0, T1)
    # print(t_loss, r_loss)
    return t_loss, r_loss


def main(hparams):    
    TOP_K = 200  # TODO: use greedy-nms for top-k to get better distributions!
    CLASS_OF_INTEREST = 4
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
        "resume": os.path.join("/home/chengyh23/Documents/CORSAIR-ERL", "ckpts", f"scannet_ret_table_best"),
        "root": "/media/sdb2/chengyh23/ShapeNetCore.v2.PCD.npy/sample15000/ncc",
        
        "embedding": "conv1_max_embedding",
        "dim": [1024, 512,  256],
        "model": "ResUNetBN2C",
        "model_n_out": 16,
        "normalize_feature": True,
        "conv1_kernel_size": 3,
        "bn_momentum": 0.05,
        
        "catid": CLS2SYNSET_ID[CLASS_OF_INTEREST],
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
        "RRE_regis": []
    }
    _i=0
    for batch in tqdm(data_generator):
        # image, seg_target, depth_target, pose_targets, detections_gt, scene_name = batch
        image, seg_target, modelIds, depth_target, pose_targets, bboxes_gt, _, img_name, scene_name = batch
        modelIds = modelIds[0]
        img_name = img_name[0]
        # if img_name != "CAMERA/val/00013/0006": continue    # DEBUG
        pred_dp = scene_grasp_model.get_predictions_from_preprocessed(image, camera_k)
        if pred_dp is None:
            print(f"[{img_name}] > No objects found.")
            continue
        print(f"[{img_name}] > ")
        
        # from scripts.demo_2d import render_predcitions
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
        from scene_grasp.scene_grasp_net.utils.matches_utils import get_matches, load_deformnet_nocs_results
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
            if pred_class_ids[pred_idx] != CLASS_OF_INTEREST:
                continue
            if gt_idx == -1:
                continue
            print("New OOI! ")
            n_ooi += 1
            # 1. Directly Predict Pose
            from utils.eval_pose import eval_pose_my
            T_est =  pred_dp.pose_matrices[pred_idx, :, :]
            T_target = nocs['gt_RTs'][gt_idx, :, :]   # 4,4
            # T_target = nocs['gt_scales'][pred_idx, :]   # 3
            t_loss1, r_loss1 = eval_pose_my(T_est, T_target)
            print('[Direct pose prediction] RTE {0:.2f}, RRE {1:.2f}'.format(t_loss1, r_loss1))
            
            # 2. Give to CORSAIR
            # base_pc = pred_dp.endpoints['xyz'][pred_idx]
            base_pc = pred_dp.obj_canonical_pcls[pred_idx]
            model_id = modelIds[gt_idx]
            synset_id = CLS2SYNSET_ID[CLASS_OF_INTEREST]
            # t_loss, r_loss = corsair_model(base_pc, config, synset_id, model_id)
            t_loss2, r_loss2 = corsair_model2(base_pc, config, synset_id, model_id, feature_extractor, retrieval_module_)
            print('[Registration-based pose prediction] RTE {0:.2f}, RRE {1:.2f}'.format(t_loss2, r_loss2))
            
            stat["img_name"].append(img_name)
            stat["pred_idx"].append(pred_idx)
            stat["gt_idx"].append(gt_idx)
            stat["RTE_pred"].append(t_loss1)
            stat["RRE_pred"].append(r_loss1)
            stat["RTE_regis"].append(t_loss2)
            stat["RRE_regis"].append(r_loss2)

        print("Total # OOI: ", n_ooi)
        _i += 1
        # if _i > 2: 
        #     break
    # Save results
    ckpt_name = os.path.split(config.resume)[1]
    catid = CLS2SYNSET_ID[CLASS_OF_INTEREST]
    now = datetime.datetime.now()
    time_str = now.strftime('%Y%m%d_%H%M%S')
    out_stat_filename = f"{ckpt_name}-{catid}-{time_str}.csv"
    df = pd.DataFrame(stat)
    df.to_csv(os.path.join("results/pose/", out_stat_filename), index=False)
    print("Write to {}".format(os.path.join("results/pose/", out_stat_filename)))
    return


if __name__ == "__main__":
    args_list = None
    if len(sys.argv) > 1:
        args_list = sys.argv[1:]
    hparams = get_scene_grasp_model_params(args_list)
    main(hparams)
