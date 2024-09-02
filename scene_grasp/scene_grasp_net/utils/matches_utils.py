""" 
Johan C.
Aug 2024

Match predictions and groundtruth. 
Based on scene_grasp/scene_grasp_net/utils/nocs_eval_utils_od.py 
"""
import os
import numpy as np
import _pickle as cPickle
import pathlib
from scene_grasp.scene_grasp_net.utils.nocs_eval_utils_od import (
    compute_IoU_matches_origin_indexed, # compute_IoU_matches,
    compute_RT_overlaps,
    compute_RT_matches
)
def load_deformnet_nocs_results(img_name, object_deformnet_nocs_results_dir):
    """_summary_

    Args:
        img_name (_type_): CAMERA/
    Return:
        dict. usage:
        gt_class_ids = nocs['gt_class_ids']
        # gt_sRT = np.array(nocs['gt_RTs'])
        gt_sRT = nocs['gt_RTs']
        gt_size = np.array(nocs['gt_scales'])
        gt_handle_visibility = nocs['gt_handle_visibility']
    """
    source = img_name.split('/')[0]
    object_deformnet_nocs_results_dir = pathlib.Path(object_deformnet_nocs_results_dir)
    nocs_dir = str(object_deformnet_nocs_results_dir.absolute())
    if source == 'CAMERA':
        nocs_path = os.path.join(nocs_dir, 'val', 'results_val_{}_{}.pkl'.format(
            img_name.split('/')[-2], img_name.split('/')[-1]))
    else:
        nocs_path = os.path.join(nocs_dir, 'real_test', 'results_test_{}_{}.pkl'.format(
            img_name.split('/')[-2], img_name.split('/')[-1]))
    with open(nocs_path, 'rb') as f:
        nocs = cPickle.load(f)
    return nocs

def get_matches(pred_dp, nocs, out_dir, degree_thresholds=[180], shift_thresholds=[100],
                iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False):
    """ Compute mean Average Precision.
    
    pred_: from pred_dp
    gt_: from result_**.pkl
    
    Returns:
        iou_pred_matches_all: (101, pred_dp.getlen())
        pose_pred_matches_all:
        iou_gt_matches_all: (101, number of objects in ground truth)
        pose_gt_matches_all:
    """
    synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)
    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)
    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    
    gt_class_ids = nocs['gt_class_ids']
    num_gt = len(gt_class_ids)
    # gt_sRT = np.array(nocs['gt_RTs'])
    gt_sRT = nocs['gt_RTs']
    gt_size = np.array(nocs['gt_scales'])
    gt_handle_visibility = nocs['gt_handle_visibility']

    # pred_class_ids = result['pred_class_ids']
    pred_class_ids = np.array(pred_dp.class_ids)    # list->numpy.ndarray: to utilize numpy's broadcast mechanism
    num_pred = len(pred_class_ids)
    # pred_sRT = np.array(result['pred_RTs'])
    pred_sRT = pred_dp.pose_matrices
    # pred_size = result['pred_scales']
    pred_size = np.diagonal(pred_dp.scale_matrices, axis1=1, axis2=2)
    pred_size = pred_size[:, :3]
    # pred_scores = result['pred_scores']
    pred_scores = np.array(pred_dp.class_confidences)   # list->numpy.ndarray

    # pre-allocate more than enough memory
    iou_pred_matches_all = np.zeros((num_iou_thres, num_pred), dtype=np.int32)
    iou_pred_scores_all = np.zeros((num_iou_thres, num_pred))
    iou_gt_matches_all = np.zeros((num_iou_thres, num_gt), dtype=np.int32)
    iou_pred_count = 0
    iou_gt_count = 0

    pose_pred_matches_all = np.zeros((num_degree_thres, num_shift_thres, num_pred), dtype=np.int32)
    pose_pred_scores_all = np.zeros((num_degree_thres, num_shift_thres, num_pred))
    pose_gt_matches_all = np.zeros((num_degree_thres, num_shift_thres, num_gt), dtype=np.int32)
    pose_pred_count = 0
    pose_gt_count = 0

    
    # print(gt_class_ids, pred_class_ids) # DEBUG
    if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
        # continue
        return iou_pred_matches_all, pose_pred_matches_all, iou_gt_matches_all, pose_gt_matches_all

    for cls_id in range(1, num_classes):
        # print("class_id, name", cls_id, synset_names[cls_id])
        # Get gt and predictions in this class
        indices_cls_gt = np.where(gt_class_ids==cls_id)[0]
        cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
        cls_gt_sRT = gt_sRT[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))
        cls_gt_size = gt_size[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
        if synset_names[cls_id] != 'mug':
            cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
        else:
            cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)

        indices_cls_pred = np.where(pred_class_ids==cls_id)[0]
        # print(cls_id, indices_cls_gt, indices_cls_pred) # DEBUG
        cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
        cls_pred_sRT = pred_sRT[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))

        # print("cls_pred_size", pred_size)
        # print("pred_size[pred_class_ids==cls_id]", pred_size[pred_class_ids==cls_id])
        cls_pred_size = pred_size[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))
        # print("pred_class_ids==cls_id", pred_class_ids==cls_id)
        # print("pred scores", pred_scores)
        # print("pred_scores[pred_class_ids==cls_id]", pred_scores[pred_class_ids==cls_id])
        cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)

        # Calculate the overlap between each gt instance and pred instance
        iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = \
            compute_IoU_matches_origin_indexed(cls_gt_class_ids, cls_gt_sRT, cls_gt_size, cls_gt_handle_visibility,
                                cls_pred_class_ids, cls_pred_sRT, cls_pred_size, cls_pred_scores,
                                synset_names, iou_thres_list)
        # if len(iou_pred_indices):
        #     cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
        #     cls_pred_sRT = cls_pred_sRT[iou_pred_indices]
        #     cls_pred_scores = cls_pred_scores[iou_pred_indices]
        # Convert indices in *_match from local (within this class) to global
        # print(iou_pred_indices)
        valid_indices = iou_cls_pred_match > -1
        iou_cls_pred_match[valid_indices] = indices_cls_gt[iou_cls_pred_match[valid_indices]]
        valid_indices = iou_cls_gt_match > -1
        iou_cls_gt_match[valid_indices] = indices_cls_pred[iou_cls_gt_match[valid_indices]]
        
        # iou_cls_pred_match = iou_cls_pred_match[iou_pred_indices]
        # iou_cls_gt_match = iou_pred_indices[iou_cls_gt_match]
        # print(cls_id, 
        #       iou_cls_gt_match[0,:] if iou_cls_gt_match.shape[-1]>0 else iou_cls_gt_match, 
        #       iou_cls_pred_match[0,:] if iou_cls_pred_match.shape[-1]>0 else iou_cls_pred_match)    # DEBUG

        num_pred = iou_cls_pred_match.shape[1]
        # pred_start = iou_pred_count
        # pred_end = pred_start + num_pred
        # iou_pred_count = pred_end
        iou_pred_matches_all[:, indices_cls_pred] = iou_cls_pred_match
        cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
        assert cls_pred_scores_tile.shape[1] == num_pred
        iou_pred_scores_all[:, indices_cls_pred] = cls_pred_scores_tile
        num_gt = iou_cls_gt_match.shape[1]
        # gt_start = iou_gt_count
        # gt_end = gt_start + num_gt
        # iou_gt_count = gt_end
        iou_gt_matches_all[:, indices_cls_gt] = iou_cls_gt_match

        if use_matches_for_pose:
            thres_ind = list(iou_thres_list).index(iou_pose_thres)
            iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]
            cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
            cls_pred_sRT = cls_pred_sRT[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
            cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
            iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
            cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
            cls_gt_sRT = cls_gt_sRT[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
            cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)

        RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_sRT, cls_gt_handle_visibility,
                                            cls_pred_class_ids, cls_pred_sRT, synset_names)
        pose_cls_gt_match, pose_cls_pred_match = compute_RT_matches(RT_overlaps, cls_pred_class_ids, cls_gt_class_ids,
                                                                    degree_thres_list, shift_thres_list)
        num_pred = pose_cls_pred_match.shape[2]
        # pred_start = pose_pred_count
        # pred_end = pred_start + num_pred
        # pose_pred_count = pred_end
        pose_pred_matches_all[:, :, indices_cls_pred] = pose_cls_pred_match
        cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
        assert cls_pred_scores_tile.shape[2] == num_pred
        pose_pred_scores_all[:, :, indices_cls_pred] = cls_pred_scores_tile
        num_gt = pose_cls_gt_match.shape[2]
        # gt_start = pose_gt_count
        # gt_end = gt_start + num_gt
        # pose_gt_count = gt_end
        pose_gt_matches_all[:, :, indices_cls_gt] = pose_cls_gt_match
    # print("===============\n")

    # # trim zeros
    # for cls_id in range(num_classes):
    #     # IoU
    #     iou_pred_matches_all = iou_pred_matches_all[:, :iou_pred_count]
    #     iou_pred_scores_all = iou_pred_scores_all[:, :iou_pred_count]
    #     iou_gt_matches_all = iou_gt_matches_all[:, :iou_gt_count]
    #     # pose
    #     pose_pred_matches_all = pose_pred_matches_all[:, :, :pose_pred_count]
    #     pose_pred_scores_all = pose_pred_scores_all[:, :, :pose_pred_count]
    #     pose_gt_matches_all = pose_gt_matches_all[:, :, :pose_gt_count]

    
    return iou_pred_matches_all, pose_pred_matches_all, iou_gt_matches_all, pose_gt_matches_all
