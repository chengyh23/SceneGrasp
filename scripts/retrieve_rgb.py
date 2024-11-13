"""
Oct 2024
Johan Cheng

Input RGB, retrieve from ShapeNet
"""
import numpy as np
from pathlib import Path
import sys
import datetime
import torch
import os
from common.utils.scene_grasp_utils import (
    SceneGraspModel,
)
from common.utils.misc_utils import (
    get_scene_grasp_model_params,
)
from scene_grasp.scene_grasp_net.utils.matches_utils import get_matches, load_deformnet_nocs_results
from scripts.class2synsetId import CLS2SYNSET_ID

CORSAIR_DIR = "/home/chengyh23/Documents/CORSAIR"
sys.path.append(CORSAIR_DIR)
from src.config import get_config
from tqdm import tqdm
import pandas as pd
import traceback

def corsair_model_retrieve(base_pc, feat_extractor, retrieval_module_):
    base_local_feat, base_global_feat, base_coords = feat_extractor.process(base_pc)
    _, topn_idx = retrieval_module_.Top1_my(base_global_feat.detach().cpu().numpy())
    
    return topn_idx

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
        try:
            # image, seg_target, depth_target, pose_targets, detections_gt, scene_name = batch
            image, seg_target, modelIds, instance_ids, depth_target, pose_targets, bboxes_gt, _, img_name, scene_name = batch
            modelIds = modelIds[0]
            img_name = img_name[0]
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
                synset_id = CLS2SYNSET_ID[class_of_interest]
                # t_loss, r_loss = corsair_model(base_pc, config, synset_id, model_id)
                # t_loss2, r_loss2 = corsair_model2(base_pc, config, synset_id, model_id, feature_extractor, retrieval_module_)
                # print('[Registration-based pose prediction] RTE {0:.2f}, RRE {1:.2f}'.format(t_loss2, r_loss2))
                
                # Refine retrieved gsplat using single-view image
                topn_idx = corsair_model_retrieve(base_pc, feature_extractor, retrieval_module_)
                retrieved_model_path = retrieval_module_.cadlib.pathes[topn_idx[0]]
                print(retrieved_model_path)
                _, retrieved_model_fname = os.path.split(retrieved_model_path)
                retrieved_model_id = retrieved_model_fname.strip('.npy')
                retrieved_gsplat_fname = f"{synset_id}-{retrieved_model_id}.ply"
                print(retrieved_gsplat_fname)
                seg_mask = pred_dp.seg_masks[pred_idx]  # (480, 640)
                black_bg = np.zeros_like(image[0])  # (4, 480, 640)
                gt_image = np.where(seg_mask, image[0], black_bg)   # (4, 480, 640)
                gt_image = np.transpose(gt_image, (1, 2, 0))
                gt_image = torch.tensor(gt_image)
                from scripts.gs import SimpleTrainer, Gaussians
                gaussians = Gaussians(ply_fname=retrieved_gsplat_fname)
                trainer = SimpleTrainer(gt_image=gt_image, gaussians=gaussians)
                trainer.train(
                    iterations=100,
                    lr=0.01,
                    save_imgs=False,
                )
                
                
                # stat["img_name"].append(img_name)
                # stat["pred_idx"].append(pred_idx)
                # stat["gt_idx"].append(gt_idx)
                # stat["RTE_pred"].append(t_loss1)
                # stat["RRE_pred"].append(r_loss1)
                # stat["RTE_regis"].append(t_loss2)
                # stat["RRE_regis"].append(r_loss2)

            print("Total # OOI: ", n_ooi)
        except Exception as e:
            print("Exception occured while processing it")
            traceback.print_exc()
            return

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