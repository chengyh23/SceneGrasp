import numpy as np
from scipy.linalg import logm, norm
CORSAIR_DIR = "/home/chengyh23/Documents/CORSAIR"
import sys
sys.path.append(CORSAIR_DIR)
from utils.visualize import Wvisualize  #CORSAIR
from utils.preprocess import apply_transform    # CORSAIR

def angle_between_transforms(matrix1, matrix2):
    # Calculate the logarithm of the difference of the matrices
    log_diff = logm(np.dot(np.linalg.inv(matrix1), matrix2))
    
    # The norm of the skew-symmetric part of the log is the angle
    angle = norm(log_diff - log_diff.T) / 2
    return angle

def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])
    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                      colors)):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)
        axis.label.set_color(c)
        axis.line.set_color(c)
        axis.set_tick_params(colors=c)
        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
        text_loc = line[1]*1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + loc[0]
        ax.text(*text_plot, axlabel.upper(), color=c,
                va="center", ha="center")
    ax.text(*offset, name, color="k", va="center", ha="center",
            bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})


def vis_tf():
    root = "/media/sdb2/chengyh23/ShapeNetCore.v2.PCD.npy/sample15000/ncc/"
    catid = "03001627"
    model_path = "03001627/val/7e7f1989e852eb41352fc7e973ba7787.npy"
    # 4356ef46fbc859a0b1f04c301b6ccc90.npy
    # 7e7f1989e852eb41352fc7e973ba7787.npy
    model_path = "02691156/val/53011f6c40df3d7b4f95630cc18536e0.npy"
    pc = np.load(root + model_path)
    T_est = np.load('T_est.npy')
    T_target2 = np.load('T_target2.npy')
    # pc_est = apply_transform(pc, T_est)
    # pc_target2 = apply_transform(pc, T_target2)
    # Wvisualize([pc, pc_est, pc_target2],["RED","BLUE","GREEN"])

    print("T_est:")
    print(T_est)
    print("T_target2:")
    print(T_target2)
    from scipy.spatial.transform import Rotation as R
    import matplotlib.pyplot as plt

    T_est = np.array(
        [[ 0.23555392,  0.06213209, -0.0036421 , -0.44847957],
        [-0.04372663,  0.17535362,  0.16339646, -0.22539528],
        [ 0.04429044, -0.1573214 ,  0.1806866 ,  1.27140901],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )
    T_target2 = np.array(
        [[ -4.58137900e-01,   1.17619568e-03,  -7.65309706e-02,  -4.32629406e-01],
        [  4.51365188e-02,  -3.70930701e-01,  -2.75901854e-01,  -2.22153202e-01],
        [ -6.18147887e-02,  -2.79567063e-01,   3.65745664e-01,   1.22866070e+00],
        [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
    )

    r_est = R.from_matrix(T_est[:3,:3])
    r_target2 = R.from_matrix(T_target2[:3,:3])

    ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
    plot_rotated_axes(ax, r_est, name="r0", offset=(0, 0, 0))
    plot_rotated_axes(ax, r_target2, name="r1", offset=(3, 0, 0))
    _ = ax.annotate(
        "r0: Identity Rotation\n"
        "r2: Extrinsic Euler Rotation (zyx)",
        xy=(0.6, 0.7), xycoords="axes fraction", ha="left"
    )
    ax.set(xlim=(-1.25, 7.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
    ax.set(xticks=range(-1, 8), yticks=[-1, 0, 1], zticks=[-1, 0, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.figure.set_size_inches(6, 5)
    plt.tight_layout()
    plt.show()

    euler_est = r_est.as_euler("zyx")
    euler_target2 = r_target2.as_euler("zyx")
    # print(euler_est, euler_target2)

    r_loss = np.arccos(
        # np.clip((np.trace(r_est.T @ r_target2) - 1) / 2, -1, 1)
        # np.clip((np.trace(T_est[:3, :3].T @ T_target2[:3, :3]) - 1) / 2, -1, 1)
        np.clip((np.trace(np.linalg.inv(T_est[:3, :3]) @ T_target2[:3, :3]) - 1) / 2, -1, 1)
    )

    Q_est,R_est = np.linalg.qr(T_est[:3, :3])
    # T_ = to_rotation_mat(T_est[:3, :3])
    Q_target2,R_target2 = np.linalg.qr(T_target2[:3, :3])
    print("Q_est:")
    print(Q_est)
    print("Q_target2:")
    print(Q_target2)
    # print(Q_est, R_est)
    # print(Q_target2,R_target2)
    r_loss = np.arccos(
        np.clip((np.trace(Q_est.T @ Q_target2) - 1) / 2, -1, 1)
    )
    print(r_loss)
    # angle = angle_between_transforms(T_est[:3,:3], T_target2[:3,:3])
    # print("Angle between the two transformation matrices:", angle)
def reg_test():
    T_reg3 = np.array([
        [ 0.99685216, -0.01485075,  0.07787967, -0.00528768],
        [ 0.06460255,  0.72158116, -0.6893092 , -0.06506548],
        [-0.04595974,  0.69217056,  0.72026914,  0.02880901],
        [ 0.        ,  0.       ,   0.       ,   1.        ],
    ])
import cv2
import numpy as np
import open3d as o3d
from common.utils.scene_grasp_utils import SceneGraspModel
from pathlib import Path
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
import torch
import os
from tqdm import tqdm
CORSAIR_DIR = "/home/chengyh23/Documents/CORSAIR"
sys.path.append(CORSAIR_DIR)
from src.config import get_config   #CORSAIR
from utils.Info.CADLib import CustomizeCADLib
from utils.visualize import Wvisualize  #CORSAIR
from utils.preprocess import apply_transform    # CORSAIR
import copy
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
        "resume": os.path.join("/home/chengyh23/Documents/CORSAIR-ERL", "ckpts", f"scannet_ret_table_best"),
        # "resume": os.path.join("/home/chengyh23/Documents/CORSAIR", "src/ckpts", "cat_ret_conv256max_01_FCGFus_can"),
        # "resume": os.path.join("/home/chengyh23/Documents/CORSAIR", "src/ckpts", "cat_ret_conv256max_01_FCGFus_laptop"),
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
        
        retrieved_pcls_shapenet_ooi_o3d = []
        retrieved_pcls_canonical_ooi_o3d = []   # shapenet -> canonical -> retrieved_pcls_ooi_o3d
        retrieved_pcls_ooi_o3d = []
        retrieved_pcls_TFframe_ooi_o3d = []
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
            
            # Retrieved (ShapeNet CAD model) pcl
            retrieved_pcls_shapenet_ooi_o3d.append(get_o3d_pcd_from_np(pos_coords, color=[1,0,0]))  # R
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh_t = copy.deepcopy(mesh).transform(T_target2)
            retrieved_pcls_TFframe_ooi_o3d.append(mesh_t)
            
            _colors = [[1,0,0],[0,1,0],[0,0,1],[0,1,1]]
            for i in range(1,4):
                T_reg = sym_pose(base_local_feat, base_coords, pos_local_feat, pos_coords, pos_sym=1)
                print(f"[{i}] T_reg:")
                print(T_reg)
                # if i==1:
                #     T_reg = np.array(
                #         [[-0.9067,  0.1474, -0.3952,  0.0593],
                #         [ 0.1590, -0.7484, -0.6439, -0.0134],
                #         [-0.3906, -0.6467,  0.6551,  0.0164],
                #         [ 0.0000,  0.0000,  0.0000,  1.0000]]
                #         )   # RRE 
                T_est = T_norm @ np.linalg.inv(T_reg)
                t_loss2, r_loss2 = eval_pose_my(T_est, T_target2)
                print('[Registration-based pose prediction] RTE {0:.2f}, RRE {1:.2f}'.format(t_loss2, r_loss2))
                pcl = apply_transform(pos_coords, np.linalg.inv(T_reg))
                retrieved_pcls_canonical_ooi_o3d.append(get_o3d_pcd_from_np(pcl, color=_colors[i]))
                retrieved_pcl3 = apply_transform(pos_coords, T_est)
                retrieved_pcl3 += [0.5*i,0,0]
                retrieved_pcls_ooi_o3d.append(get_o3d_pcd_from_np(retrieved_pcl3, color=_colors[i]))
                mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                T_est[:3,3] += [0.5*i,0,0]
                mesh_t = copy.deepcopy(mesh).transform(T_est)
                retrieved_pcls_TFframe_ooi_o3d.append(mesh_t)
            
            
        # # Visualize pred pcls & retrieved pcls
        # retrieved_pcls_shapenet_ooi_o3d = []
        # retrieved_pcls_canonical_ooi_o3d = []   # shapenet -> canonical -> retrieved_pcls_ooi_o3d
        
        # for idx, can_idx in enumerate(pred_ooi_indices):
            
        #     # Retrieved (ShapeNet CAD model) pcl
        #     retrieved_pcls_shapenet_ooi_o3d.append(get_o3d_pcd_from_np(retrieved_pcls_ooi[idx], color=[1,0,0]))
            
        #     pcl = apply_transform(retrieved_pcls_ooi[idx], np.linalg.inv(retrieved_pcls_ooi_T_reg[idx]))
        #     retrieved_pcls_canonical_ooi_o3d.append(get_o3d_pcd_from_np(pcl, color=[0.5,0,0.5]))

            
        
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
                # + retrieved_pcls_shapenet_ooi_o3d + retrieved_pcls_canonical_ooi_o3d \
                + retrieved_pcls_ooi_o3d + retrieved_pcls_TFframe_ooi_o3d
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