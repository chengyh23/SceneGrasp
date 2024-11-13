"""
Oct 2024
Johan Cheng

Visualization/Debug for retrieve_rgb.py
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import datetime
import torch
import os
from common.utils.scene_grasp_utils import (
    SceneGraspModel,
)
from common.utils.nocs_utils import load_depth
from common.utils.misc_utils import (
    convert_realsense_rgb_depth_to_o3d_pcl,
    get_o3d_pcd_from_np,
    get_scene_grasp_model_params,
)
from scene_grasp.scene_grasp_net.utils.matches_utils import get_matches, load_deformnet_nocs_results
from scripts.class2synsetId import CLS2SYNSET_ID
from scripts.rgbd2pcls import get_object_pc_from_scene
from scripts.demo_compare import normalize_pc

from argparse import ArgumentParser
# SHAPESPLAT_DIR = "/home/chengyh23/Documents/ShapeSplat-Gaussian_MAE"
# sys.path.append(SHAPESPLAT_DIR)
from gaussian_utils.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_utils.scene.gaussian_model import GaussianModel
from gaussian_utils.gs_train import training

CORSAIR_DIR = "/home/chengyh23/Documents/CORSAIR"
sys.path.append(CORSAIR_DIR)
from src.config import get_config
from utils.preprocess import apply_transform    # CORSAIR
from tqdm import tqdm
import pandas as pd
import traceback
import open3d as o3d
# ---Transform ShapeSplat to ShapeNet---
# xz90 = np.array(
#     [[0, -1, 0],
#     [0, 0, -1],
#     [1, 0, 0]]
# )
xz90 = np.array(
    [[0, -1, 0],
    [0, 0, 1],
    [-1, 0, 0]]
)
T_xz90 = np.eye(4)
T_xz90[:3, :3] = xz90
# --------------------------------------

def corsair_model_retrieve(base_pc, feat_extractor, retrieval_module_):
    base_local_feat, base_global_feat, base_coords = feat_extractor.process(base_pc)
    _, topn_idx = retrieval_module_.Top1_my(base_global_feat.detach().cpu().numpy())
    
    return topn_idx

def corsair_model_retreg(base_pc, feat_extractor, retrieval_module_):
    base_local_feat, base_global_feat, base_coords = feat_extractor.process(base_pc)
    _, topn_idx = retrieval_module_.Top1_my(base_global_feat.detach().cpu().numpy())
    
    pos_local_feat = retrieval_module_.local_feat_lib[topn_idx[0]]
    pos_coords = retrieval_module_.cadlib[topn_idx[0]]["origin"]
    # Registration using local feature
    from src.scene_level import sym_pose    # CORSAIR
    from utils.eval_pose import eval_pose   # CORSAIR
    T_reg = sym_pose(base_local_feat, base_coords, pos_local_feat, pos_coords, pos_sym=1)
    T_reg = T_reg.numpy()
    print("T_reg:")
    print(T_reg)
    return topn_idx, T_reg
    
    
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
        "RRE_regis": []
    }
    _i=0
    print("begin from DATA GENERATOR")
    for batch in tqdm(data_generator):
        _i += 1
        if _i > 500: 
            break
        
        # image, seg_target, depth_target, pose_targets, detections_gt, scene_name = batch
        image, seg_target, modelIds, instance_ids, depth_target, pose_targets, bboxes_gt, _, img_name, scene_name = batch
        modelIds = modelIds[0]
        img_name = img_name[0]
        # Read original image
        image_color_path = os.path.join("/media/sdb2/chengyh23/NOCS", img_name+"_color.png")
        image_color_origin = Image.open(image_color_path)
        image_color_origin = np.array(image_color_origin)

        # if img_name != "CAMERA/val/02159/0000": continue    # DEBUG
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
        from scene_grasp.scene_grasp_net.data_generation.generate_data_nocs import process_data
        img_full_path = os.path.join("data/NOCSDataset", img_name)
        depth_full_path_real = img_full_path + '_depth.png'
        depth = load_depth(depth_full_path_real)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        # assert pred_dp.get_len() == len(modelIds)
        # for i in range(pred_dp.get_len()):
            # Filter class of interest AND instance matched well with ground truth
        #     class_id = pred_dp.class_ids[i]
        #     synset_id = CLS2SYNSET_ID[class_id]
        #     if synset_id != "03642806":
        #         continue
        #     if iou_pred_matches_all[i] == -1:
        #         continue
        
        pred_ooi_indices = []
        retrieved_pcls_ooi = []
        retrieved_gsplat_ooi = []
        refined_gsplat_ooi = []
        retrieved_pcls_ooi_T_target = []
        
        n_ooi = 0   # number of objects of interest
        for pred_idx, gt_idx in enumerate(iou_pred_matches_all[0, :]):
            # Filter class of interest AND instance matched well with ground truth
            if pred_class_ids[pred_idx] != class_of_interest:
                continue
            if gt_idx == -1:
                continue
            print("New OOI! ")
            n_ooi += 1
            
            # 1. Directly Predict Pose
            pred_ooi_indices.append(pred_idx)
            from utils.eval_pose import eval_pose_my
            T_est =  pred_dp.pose_matrices[pred_idx, :, :]
            T_target = nocs['gt_RTs'][gt_idx, :, :]   # 4,4
            # T_target = nocs['gt_scales'][pred_idx, :]   # 3
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
            t_loss1, r_loss1 = eval_pose_my(T_est, T_target)
            print('[Direct pose prediction] RTE {0:.2f}, RRE {1:.2f}'.format(t_loss1, r_loss1))
            
            # 2. Give to CORSAIR
            
            # # 2.1 Input SceneGrasp's prediction to CORSAIR
            # # base_pc = pred_dp.endpoints['xyz'][pred_idx]
            # base_pc = pred_dp.obj_canonical_pcls[pred_idx]
            # # t_loss, r_loss = corsair_model(base_pc, config, synset_id, model_id)
            # # t_loss2, r_loss2 = corsair_model2(base_pc, config, synset_id, model_id, feature_extractor, retrieval_module_)
            # # print('[Registration-based pose prediction] RTE {0:.2f}, RRE {1:.2f}'.format(t_loss2, r_loss2))
            # # ----------------------------2.1
            
            # 2.2 Input scene scan to CORSAIR
            print("HEY!!", instance_ids, instance_ids[gt_idx])
            base_pc = get_object_pc_from_scene(img_name, instance_ids[gt_idx])
            pc_center, r, offset = normalize_pc(base_pc)
            T_norm = np.eye(4)
            T_norm[0:3, 0:3] = r * np.eye(3)
            T_norm[0:3, 3] = offset
            # ----------------------------2.2
            model_id = modelIds[gt_idx]
            synset_id = CLS2SYNSET_ID[class_of_interest]
            
            # topn_idx = corsair_model_retrieve(base_pc, feature_extractor, retrieval_module_)
            topn_idx, T_reg = corsair_model_retreg(base_pc, feature_extractor, retrieval_module_)
            T_est = T_norm @ np.linalg.inv(T_reg)   # 2.2
            pos_coords = retrieval_module_.cadlib[topn_idx[0]]["origin"]
            retrieved_pcls_ooi.append(pos_coords)
            
            # 3. Refine retrieved gsplat using single-view image
            retrieved_model_path = retrieval_module_.cadlib.pathes[topn_idx[0]]
            print(retrieved_model_path)
            _, retrieved_model_fname = os.path.split(retrieved_model_path)
            retrieved_model_id = retrieved_model_fname.strip('.npy')
            retrieved_gsplat_fname = f"{synset_id}-{retrieved_model_id}.ply"
            print(retrieved_gsplat_fname)
            # Mask object instance
            # 1) Get segmentation mask
            # # 1.a) use pred
            # seg_mask = pred_dp.seg_masks[pred_idx]  # (480, 640)
            # seg_mask_expanded = np.expand_dims(seg_mask, axis=-1)  # (480, 640, 1)
            # 1.b) use ground truth
            mask_inst = masks[:,:,gt_idx]
            seg_mask_expanded = np.expand_dims(mask_inst, axis=2)
            # 2) Mask image using seg_mask
            # # 2.a) using preprocessed image in batch
            # black_bg = np.zeros_like(image[0])  # (4, 480, 640)
            # gt_image = np.where(seg_mask, image[0], black_bg)   # (4, 480, 640)
            # gt_image = np.transpose(gt_image, (1, 2, 0))
            # 2.b) using original image
            black_bg = np.zeros_like(image_color_origin)  # (480, 640, 3)
            white_bg = np.ones_like(image_color_origin)  # (480, 640, 3)
            gt_image = np.where(seg_mask_expanded, image_color_origin, black_bg)   # (480, 640, 3)
            gt_image_normalized = gt_image / 255.0   # Normalize to [0,1], to fit gsplpat
            gt_image_normalized = torch.tensor(gt_image_normalized, dtype=torch.float32)
            from scripts.gs import SimpleTrainer, Gaussians
            # gaussians = Gaussians(ply_fname=retrieved_gsplat_fname)
            gaussians = GaussianModel(sh_degree=3)
            model_root_path = "/media/sdb2/chengyh23/ShapeSplatsV1/gs_data/"
            gaussians.load_ply(os.path.join(model_root_path, retrieved_gsplat_fname))
            retrieved_gsplat_ooi.append(gaussians.get_xyz.detach().cpu().numpy())
            # trainer = SimpleTrainer(gt_image=gt_image, gaussians=gaussians)
            # trainer.train(
            #     iterations=100,
            #     lr=0.01,
            #     save_imgs=False,
            # )
            parser = ArgumentParser(description="Training script parameters")
            op = OptimizationParams(parser)
            from gaussian_utils.gs_train import training
            # w2c = T_est
            w2c = T_target2 @ T_xz90
            training(op, gaussians, w2c, gt_image_normalized, iterations=400, save_imgs=True)
            refined_gsplat_ooi.append(gaussians.get_xyz.detach().cpu().numpy())
            
            retrieved_pcls_ooi_T_target.append(T_target2)
            # stat["img_name"].append(img_name)
            # stat["pred_idx"].append(pred_idx)
            # stat["gt_idx"].append(gt_idx)
            # stat["RTE_pred"].append(t_loss1)
            # stat["RRE_pred"].append(r_loss1)
            # stat["RTE_regis"].append(t_loss2)
            # stat["RRE_regis"].append(r_loss2)
        if n_ooi ==0: 
            continue
        
        return
        
        # Visualize pred pcls & retrieved pcls/gsplats
        retrieved_pcls_shapenet_ooi_o3d = []
        retrieved_gsplats_ooi_o3d = []
        # retrieved_gsplats_nocs_ooi_o3d = []
        retrieved_gsplats_nocs_target_ooi_o3d = []
        refined_gsplats_ooi_o3d = []
        refined_gsplats_nocs_target_ooi_o3d = []
        for idx, can_idx in enumerate(pred_ooi_indices):
            # Retrieved (ShapeNet CAD model) pcl
            retrieved_pcls_shapenet_ooi_o3d.append(get_o3d_pcd_from_np(retrieved_pcls_ooi[idx], color=[1,0,0]))
            # Retrieved ShapeSplat
            retrieved_gsplats_ooi_o3d.append(get_o3d_pcd_from_np(retrieved_gsplat_ooi[idx], color=[0,0,0]))
            
            retrieved_gsplat_target2 = apply_transform(retrieved_gsplat_ooi[idx], T_xz90)
            retrieved_gsplat_target3 = apply_transform(retrieved_gsplat_target2, retrieved_pcls_ooi_T_target[idx])
            retrieved_gsplats_nocs_target_ooi_o3d.append(get_o3d_pcd_from_np(retrieved_gsplat_target3, color=[1,1,0]))    # yellow
            
            # Refined ShapeSplat
            refined_gsplats_ooi_o3d.append(get_o3d_pcd_from_np(refined_gsplat_ooi[idx], color=[0,0,1])) # blue
            
            refined_gsplat_target2 = apply_transform(refined_gsplat_ooi[idx], T_xz90)
            refined_gsplat_target3 = apply_transform(refined_gsplat_target2, retrieved_pcls_ooi_T_target[idx])
            # refined_pcls_nocs_ooi_o3d.append(get_o3d_pcd_from_np(refined_gsplat_target2, color=[0,0.5,0]))    # green
            refined_gsplats_nocs_target_ooi_o3d.append(get_o3d_pcd_from_np(refined_gsplat_target3, color=[0,1,0]))    # green
            
            
        # background
        color_img_path = os.path.join("data/NOCSDataset/", img_name + "_color.png")
        depth_img_path = os.path.join("data/NOCSDataset/", img_name + "_depth.png")
        color_img = cv2.imread(str(color_img_path))  # type:ignore
        depth_img = load_depth(str(depth_img_path))
        rgb, depth = color_img, depth_img
        o3d_pcl = convert_realsense_rgb_depth_to_o3d_pcl(rgb, depth / 1000, camera_k)
        o3d.visualization.webrtc_server.enable_webrtc()
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        o3d.visualization.draw(  # type:ignore
            [coordinate_frame] \
            + [o3d_pcl] \
                # + retrieved_pcls_shapenet_ooi_o3d \
                # + retrieved_gsplats_ooi_o3d \
                + retrieved_gsplats_nocs_target_ooi_o3d \
                # + refined_gsplats_ooi_o3d \
                + refined_gsplats_nocs_target_ooi_o3d \
        )
    
        print("Total # OOI: ", n_ooi)

    # # Save results
    # ckpt_name = os.path.split(config.resume)[1]
    # catid = CLS2SYNSET_ID[class_of_interest]
    # now = datetime.datetime.now()
    # time_str = now.strftime('%Y%m%d_%H%M%S')
    # out_stat_filename = f"{ckpt_name}-{catid}-{time_str}.csv"
    # df = pd.DataFrame(stat)
    # df.to_csv(os.path.join("results/pose/", out_stat_filename), index=False)
    # print("Write to {}".format(os.path.join("results/pose/", out_stat_filename)))

    return

if __name__ == "__main__":
    # Write stat
    args_list = None
    if len(sys.argv) > 1:
        args_list = sys.argv[1:]
    hparams = get_scene_grasp_model_params(args_list)
    main(hparams)