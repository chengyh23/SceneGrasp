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
from common.utils.misc_utils import transform_pcl
from common.utils.scene_grasp_utils import (
    SceneGraspModel,
    get_final_grasps_from_predictions_np,
    get_grasp_vis,
)
from scene_grasp.scene_grasp_net.utils.matches_utils import get_matches, load_deformnet_nocs_results
from scripts.class2synsetId import CLS2SYNSET_ID
from scripts.rgbd2pcls import get_object_pc_from_scene

import torch
import os
from tqdm import tqdm
CORSAIR_DIR = "/home/chengyh23/Documents/CORSAIR"
sys.path.append(CORSAIR_DIR)
from src.config import get_config   #CORSAIR
from utils.Info.CADLib import CustomizeCADLib
from utils.visualize import Wvisualize  #CORSAIR
from utils.preprocess import apply_transform    # CORSAIR

def get_camera_frame_pcls(obj_canonical_pcls, dp):
    camera_frame_pcls = []
    for can_idx, can_pcl in enumerate(obj_canonical_pcls):
        scale_matrix = dp.scale_matrices[can_idx]
        pose_matrix = dp.pose_matrices[can_idx]
        camera_pcl = transform_pcl(can_pcl, pose_matrix, scale_matrix)
        camera_frame_pcls.append(camera_pcl)
    return camera_frame_pcls

def normalize_pc(pc):
    # Normalize point clouds of scan objects
    offset = pc.mean(0)
    pc_center = pc - offset
    r = np.linalg.norm(pc_center, 2, 1).max()
    pc_center /= r
    return pc_center, r, offset
    
def main(hparams, ):
    """
    Qualitative comparison between SceneGrasp and CORSAIR
    
    Ref. 
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
        
    from src.test_time.FeatureExtractor import FeatureExtractor # CORSAIR
    feature_extractor = FeatureExtractor(config)
    from src.test_time.RetrievalModule import RetrievalModule   # CORSAIR
    retrieval_module_ = RetrievalModule(config, feature_extractor, "shapenet", update=True)

    _i=0
    for batch in tqdm(data_generator):
        _i += 1
        # if _i > 3000: 
        #     break
        
        # image, seg_target, depth_target, pose_targets, detections_gt, scene_name = batch
        image, seg_target, modelIds, instance_ids, depth_target, pose_targets, bboxes_gt, _, img_name, scene_name = batch
        modelIds = modelIds[0]
        img_name = img_name[0]
        # if img_name != "CAMERA/val/01092/0002": continue    # DEBUG
        # if img_name != "CAMERA/val/00535/0007": continue    # DEBUG
        if img_name != "CAMERA/val/00337/0008": continue    # DEBUG
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
        print("matches: ", iou_gt_matches_all[0,:], iou_pred_matches_all[0,:])   # DEBUG
        
        # >>>>>> Compare direct pose prediction VS RANSAC pose estimation >>>>>>
        from scripts.class2synsetId import WORD2CLS_ID, WORD2SYNSET_ID
        # assert pred_dp.get_len() == len(modelIds)
        from scene_grasp.scene_grasp_net.data_generation.generate_data_nocs import process_data
        img_full_path = os.path.join("data/NOCSDataset", img_name)
        depth_full_path_real = img_full_path + '_depth.png'
        depth = load_depth(depth_full_path_real)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        coord_path = img_full_path + '_coord.png'
        coord_map = cv2.imread(coord_path)[:, :, :3]
        coord_map = coord_map[:,:,(2,1,0)]
        coord_map = np.array(coord_map, dtype=np.float32) / 255

        query_pcls_ooi = []
        query_pcls_norm_ooi = []
        # pred_pcls_ooi = []
        pred_ooi_indices = []
        nocs_ooi = []
        retrieved_pcls_ooi = []
        retrieved_pcls_ooi_T_reg = []
        retrieved_pcls_ooi_r = []
        retrieved_pcls_ooi_offset = []
        retrieved_pcls_ooi_T_norm = []
        retrieved_pcls_ooi_T_est = []
        retrieved_pcls_ooi_T_target = []
        ooi_T_target = []
        n_ooi = 0   # number of objects of interest
        for pred_idx, gt_idx in enumerate(iou_pred_matches_all[0, :]):
            # Filter class of interest AND instance matched well with ground truth
            if pred_class_ids[pred_idx] != class_of_interest:
                continue
            if gt_idx == -1:
                continue
            print("New OOI! ")
            n_ooi += 1
            # 0. NOCS map
            mask_inst = masks[:,:,gt_idx]
            mask_inst = np.expand_dims(mask_inst, axis=2)
            cd1 = mask_inst * coord_map
            
            # cd1[:, :, 2] = 1 - cd1[:, :, 2]
            # cd1 -= [0.5,0.5,0.5]
            # cd1 = cd1[:, :, (2,1,0)]
            cd1 -= [0.5,0.5,0.5]
            cd1_pc = cd1[np.any(cd1!= [0, 0, 0], axis=2)]
            nocs_ooi.append(cd1_pc)
            # 1. Directly Predict Pose
            pred_ooi_indices.append(pred_idx)
            from utils.eval_pose import eval_pose_my    #CORSAIR
            T_est =  pred_dp.pose_matrices[pred_idx, :, :]
            T_target = nocs['gt_RTs'][gt_idx, :, :]   # 4,4
            # T_target = nocs['gt_scales'][pred_idx, :]   # 3
            ooi_T_target.append(T_target)
            print("det", np.linalg.det(T_est), np.linalg.det(T_target))
            t_loss1, r_loss1 = eval_pose_my(T_est, T_target)
            print('[Direct pose prediction] RTE {0:.2f}, RRE {1:.2f}'.format(t_loss1, r_loss1))
            # 2. Give to CORSAIR
            # # base_pc = pred_dp.endpoints['xyz'][pred_idx]
            # base_pc = pred_dp.obj_canonical_pcls[pred_idx]
            # base_pc = transform_pcl(base_pc, np.eye(4), pred_dp.scale_matrices[pred_idx])   # prediction to Canoincal Coordinates
            base_pc = get_object_pc_from_scene(img_name, instance_ids[gt_idx])
            pc_center, r, offset = normalize_pc(base_pc)
            query_pcls_ooi.append(base_pc)
            query_pcls_norm_ooi.append(pc_center)
            T_norm = np.eye(4)
            T_norm[0:3, 0:3] = r * np.eye(3)
            T_norm[0:3, 3] = offset
            retrieved_pcls_ooi_r.append(r)
            retrieved_pcls_ooi_offset.append(offset)
            retrieved_pcls_ooi_T_norm.append(T_norm)
            # pred_pcls_ooi.append(base_pc)
            model_id = modelIds[gt_idx]
            synset_id = CLS2SYNSET_ID[class_of_interest]
            
            base_local_feat, base_global_feat, base_coords = feature_extractor.process(pc_center)
            _, topn_idx = retrieval_module_.Top1_my(base_global_feat.detach().cpu().numpy())
            pos_local_feat = retrieval_module_.local_feat_lib[topn_idx[0]]
            pos_coords = retrieval_module_.cadlib[topn_idx[0]]["origin"]
            pos_coord_grid = retrieval_module_.cadlib[topn_idx[0]]["coord"]
            T1 = retrieval_module_.cadlib[topn_idx[0]]["T"]
            # Registration using local feature
            from src.scene_level import sym_pose    # CORSAIR
            from utils.eval_pose import eval_pose   # CORSAIR
            T_reg = sym_pose(base_local_feat, base_coords, pos_local_feat, pos_coords, pos_sym=1)
            T_reg = T_reg.numpy()
            print("T_reg:")
            print(T_reg)
            T_est = T_norm @ np.linalg.inv(T_reg)
            retrieved_pcls_ooi.append(pos_coords)
            retrieved_pcls_ooi_T_reg.append(T_reg)
            retrieved_pcls_ooi_T_est.append(T_est)
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
            retrieved_pcls_ooi_T_target.append(T_target2)
            print("det", np.linalg.det(T_est), np.linalg.det(T_target2))
            # print("T_est:")
            # print(T_est)
            # print("T_target2:")
            # print(T_target2)
            t_loss2, r_loss2 = eval_pose_my(T_est, T_target2)
            print('[Registration-based pose prediction] RTE {0:.2f}, RRE {1:.2f}'.format(t_loss2, r_loss2))
            
            
        # Visualize pred pcls & retrieved pcls
        query_pcls_norm_ooi_o3d = []
        nocs_pcls_ooi_o3d = []
        nocs_pcls_target_ooi_o3d = []
        pred_canonical_pcls = pred_dp.obj_canonical_pcls
        pred_pcls = pred_dp.get_camera_frame_pcls()
        pred_canonical_pcls_ooi_3d = []
        pred_pcls_ooi_o3d = []
        retrieved_pcls_shapenet_ooi_o3d = []
        retrieved_pcls_canonical_ooi_o3d = []   # shapenet -> canonical -> retrieved_pcls_ooi_o3d
        retrieved_pcls_ooi_o3d = []
        retrieved_pcls_nocs_ooi_o3d = []    # shapenet -> nocs -> target
        retrieved_pcls_nocs_target_ooi_o3d = []
        for idx, can_idx in enumerate(pred_ooi_indices):
            query_pcls_norm_ooi_o3d.append(get_o3d_pcd_from_np(query_pcls_norm_ooi[idx], color=[0.2,0.2,0.2]))
            # NOCS pcl
            nocs_pcls_ooi_o3d.append(get_o3d_pcd_from_np(nocs_ooi[idx], color=[1,0.3,0]))   # orange
            nocs_target_pcl = transform_pcl(nocs_ooi[idx], pred_dp.pose_matrices[can_idx], pred_dp.scale_matrices[can_idx])   # 这是人家pred出来的
            nocs_target_pcl = apply_transform(nocs_ooi[idx], ooi_T_target[idx]) # gt
            nocs_pcls_target_ooi_o3d.append(get_o3d_pcd_from_np(nocs_target_pcl, color=[1,0.7,0]))   # orange
            
            # Pred pcl
            pred_canonical_pcls_ooi_3d.append(get_o3d_pcd_from_np(pred_canonical_pcls[can_idx], color=[0,0,1]))
            pred_pcls_ooi_o3d.append(get_o3d_pcd_from_np(pred_pcls[can_idx]))
            
            # Retrieved (ShapeNet CAD model) pcl
            retrieved_pcls_shapenet_ooi_o3d.append(get_o3d_pcd_from_np(retrieved_pcls_ooi[idx], color=[1,0,0]))
            
            pcl = apply_transform(retrieved_pcls_ooi[idx], np.linalg.inv(retrieved_pcls_ooi_T_reg[idx]))
            retrieved_pcls_canonical_ooi_o3d.append(get_o3d_pcd_from_np(pcl, color=[0.5,0,0.5]))

            # scale_matrix = pred_dp.scale_matrices[can_idx]
            # pose_matrix = pred_dp.pose_matrices[can_idx]
            # retrieved_pcl = transform_pcl(pcl, pose_matrix, scale_matrix)   # 这是人家pred出来的
            retrieved_pcl = pcl * retrieved_pcls_ooi_r[idx] + retrieved_pcls_ooi_offset[idx]
            # retrieved_pcl2 = apply_transform(pcl, retrieved_pcls_ooi_T_norm[idx])
            # assert np.all(retrieved_pcl==retrieved_pcl2), "In-equivalent trasformation"
            retrieved_pcl2 = apply_transform(retrieved_pcls_ooi[idx], retrieved_pcls_ooi_T_norm[idx])
            retrieved_pcl3 = apply_transform(retrieved_pcls_ooi[idx], retrieved_pcls_ooi_T_est[idx])
            retrieved_pcls_ooi_o3d.append(get_o3d_pcd_from_np(retrieved_pcl3, color=[1,0,1]))   # magenta
            # pred_pcls[can_idx] & query_pcls_ooi[idx]    # TODO Use Chamfer distance as metric?
            # retrieved_pcl3 & query_pcls_ooi[idx]    # TODO Use Chamfer distance as metric?
            from common.utils.misc_utils import get_o3d_chamfer_distance
            cd_pred = get_o3d_chamfer_distance(
                get_o3d_pcd_from_np(pred_pcls[can_idx], color=[1,0,1]), 
                get_o3d_pcd_from_np(query_pcls_ooi[idx], color=[0.2,0.2,0.2])
                )
            cd_retrieved = get_o3d_chamfer_distance(
                get_o3d_pcd_from_np(retrieved_pcl3, color=[1,0,1]), 
                get_o3d_pcd_from_np(query_pcls_ooi[idx], color=[0.2,0.2,0.2])
                )
            print(f"cd pred: {cd_pred}; cd CORSAIR: {cd_retrieved}")
            # Retrieved (ShapeNet) -> NOCS
            from scripts.rgbd2pcls import ShapeNet2PredCanonical
            pcl_nocs = ShapeNet2PredCanonical(retrieved_pcls_ooi[idx])
            # pcl_nocs = retrieved_pcls_ooi[can_idx]
            # pcl_nocs[:, 2] = - pcl_nocs[:, 2]
            # pcl_nocs = pcl_nocs[:, (2,1,0)]
            # # pcl_nocs += [0.5,0.5,0.5]
            pcl_nocs2 = apply_transform(retrieved_pcls_ooi[idx], T_ShapeNet2PredCanonical)
            assert np.all(pcl_nocs==pcl_nocs2), "In-equivalent trasformation"
            retrieved_pcls_nocs_ooi_o3d.append(get_o3d_pcd_from_np(pcl_nocs, color=[0,0.5,0]))
            retrieved_pcl_target = apply_transform(pcl_nocs, ooi_T_target[idx])
            retrieved_pcl_target2 = apply_transform(retrieved_pcls_ooi[idx], ooi_T_target[idx] @ T_ShapeNet2PredCanonical)
            retrieved_pcl_target3 = apply_transform(retrieved_pcls_ooi[idx], retrieved_pcls_ooi_T_target[idx])
            retrieved_pcls_nocs_target_ooi_o3d.append(get_o3d_pcd_from_np(retrieved_pcl_target3, color=[0,1,0]))    # green
        
        # background
        color_img_path = os.path.join("data/NOCSDataset/", img_name + "_color.png")
        depth_img_path = os.path.join("data/NOCSDataset/", img_name + "_depth.png")
        color_img = cv2.imread(str(color_img_path))  # type:ignore
        depth_img = load_depth(str(depth_img_path))
        rgb, depth = color_img, depth_img
        o3d_pcl = convert_realsense_rgb_depth_to_o3d_pcl(rgb, depth / 1000, camera_k)
        o3d.visualization.webrtc_server.enable_webrtc()
        print(">Showing predicted shapes:")
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        o3d.visualization.draw(  # type:ignore
            [coordinate_frame] + \
            [o3d_pcl] \
                + pred_canonical_pcls_ooi_3d + pred_pcls_ooi_o3d \
                + retrieved_pcls_shapenet_ooi_o3d \
                + retrieved_pcls_canonical_ooi_o3d + retrieved_pcls_ooi_o3d \
                # + retrieved_pcls_nocs_ooi_o3d + retrieved_pcls_nocs_target_ooi_o3d \
                # + nocs_pcls_ooi_o3d + nocs_pcls_target_ooi_o3d
                # + query_pcls_norm_ooi_o3d \
        )

        print("Total # OOI: ", n_ooi)
        # except Exception as e:
        #     print("Exception occured while processing it")
        #     print("\t{}: {}".format(type(e).__name__, e)) 
        #     raise 
    return


if __name__ == "__main__":
    args_list = None
    if len(sys.argv) > 1:
        args_list = sys.argv[1:]
    hparams = get_scene_grasp_model_params(args_list)
    main(hparams)

