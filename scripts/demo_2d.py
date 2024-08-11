"""
Aug-11 2024
Johan Cheng

This script is used to demonstrate object detection results on 2D images.

Ref. CS_Inference::save_predictions() in common/utils/cs_inference_utils.py
"""

from pathlib import Path
import sys
import numpy as np
from common.utils.misc_utils import (
    get_scene_grasp_model_params,
)
from common.utils.scene_grasp_utils import (
    SceneGraspModel,
)
from demo import get_demo_data_generator


import os
import cv2
import open3d as o3d
from simnet.lib import camera
# from scene_grasp.scene_grasp_net.utils.nocs_eval_utils import draw_bboxes
from scene_grasp.scene_grasp_net.utils.viz_utils import draw_bboxes, draw_bboxes_2d
from scene_grasp.scene_grasp_net.utils.transform_utils import (
    get_gt_pointclouds,
    transform_coordinates_3d,
    calculate_2d_projections,
)
from scene_grasp.scene_grasp_net.utils.transform_utils import project


def main(hparams):
    TOP_K = 200  # TODO: use greedy-nms for top-k to get better distributions!
    # Model:
    print("Loading model from checkpoint: ", hparams.checkpoint)
    scene_grasp_model = SceneGraspModel(hparams)

    # demo_data_path = Path("outreach/demo_data")
    # data_generator = get_demo_data_generator(demo_data_path)
    # for rgb, depth, camera_k in data_generator:
    #     print("------- Showing results ------------")
    #     pred_dp = scene_grasp_model.get_predictions_2d(rgb, depth, camera_k, output_path=Path("results/CenterSnap"))
    #     if pred_dp is None:
    #         print("No objects found.")
    #         continue
    data_generator = scene_grasp_model.model.val_dataloader()
    demo_data_path = Path("outreach/demo_data")
    camera_k = np.loadtxt(demo_data_path / "camera_k.txt")
    n = 0
    for batch in data_generator:
        print("------- Showing results ------------")
        image, seg_target, modelIds, depth_target, pose_targets, bboxes_gt, detections_gt, img_name, scene_name = batch
        # pred_dp = scene_grasp_model.get_predictions_2d_from_preprocessed(image, camera_k, img_vis=img_name, bboxes_gt=bboxes_gt, output_path=Path("results/CenterSnap"),)
        pred_dp = scene_grasp_model.get_predictions_from_preprocessed(image, camera_k)
        # , img_vis=img_name, bboxes_gt=bboxes_gt, output_path=Path("results/CenterSnap"),)
        if pred_dp is None:
            print("No objects found.")
            continue
        
        # batch
        #     img_vis: list[str] or numpy array (480, 640, 3)
        #     bboxes_gt: list[numpy array (num_inst, 4)]
        # =================== Visualization ====================
        output_path=Path("results/CenterSnap")
        scale_depth = False
        write_pcd = False
        
        # Load Background Image
        img_vis=img_name
        if isinstance(img_vis, list):
            img_vis = img_vis[0]
            assert isinstance(img_vis, str)
            output_prefix = img_vis.replace('/','-')
            img_full_path= os.path.join("data/NOCSDataset", img_vis)
            color_img_path = img_full_path + '_color.png'
            if not os.path.exists(color_img_path):
                raise ValueError
            img_vis = cv2.imread(color_img_path)  # type:ignore
        
        pred_xyzs = pred_dp.endpoints["xyz"]
        
        shape_out = pred_xyzs
        
        # write_pcd = True
        cam_pcls = []
        points_2d = []
        box_obb = []
        axes = []

        for j in range(pred_dp.pose_matrices.shape[0]):
            # _, shape_out = self.cs_ae(None, emb)
            # shape_out = shape_out.cpu().detach().numpy()[0]
            from simnet.lib import transform
            abs_pose_output = transform.Pose(
                camera_T_object=pred_dp.pose_matrices[j, :, :],
                scale_matrix=pred_dp.scale_matrices[j, :, :]
            )
            rotated_pc, rotated_box, _ = get_gt_pointclouds(
                abs_pose_output, shape_out[j], camera_model=None
            )
            if scale_depth:
                rotated_pc = rotated_pc / scale_ratio
            cam_pcls.append(rotated_pc)
            if write_pcd:
                ply_path = output_path / f"{output_prefix}_obj_{j}_cam_pcl.ply"
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(rotated_pc)
                o3d.io.write_point_cloud(str(ply_path), pcd)

            # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            #     size=0.1, origin=[0, 0, 0]
            # )
            # T = abs_pose_outputs[j].camera_T_object
            # mesh_frame = mesh_frame.transform(T)
            # cam_pcls.append(mesh_frame)
            # cylinder_segments = line_set_mesh(rotated_box)
            # for k in range(len(cylinder_segments)):
            #     cam_pcls.append(cylinder_segments[k])

            points_mesh = camera.convert_points_to_homopoints(rotated_pc.T)
            points_2d_mesh = project(camera_k, points_mesh)
            points_2d_mesh = points_2d_mesh.T
            points_2d.append(points_2d_mesh)
            # 2D output
            points_obb = camera.convert_points_to_homopoints(np.array(rotated_box).T)
            points_2d_obb = project(camera_k, points_obb)
            points_2d_obb = points_2d_obb.T
            box_obb.append(points_2d_obb)
            xyz_axis = (
                0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            )
            sRT = abs_pose_output.camera_T_object @ abs_pose_output.scale_matrix
            transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
            projected_axes = calculate_2d_projections(
                transformed_axes, camera_k[:3, :3]
            )
            axes.append(projected_axes)
        colors_box = [(63, 237, 234)]
        im = np.array(np.copy(img_vis)).copy()
        for k in range(len(colors_box)):
            for points_2d, axis in zip(box_obb, axes):
                points_2d = np.array(points_2d)
                print("draw_bboxes")
                im = draw_bboxes(im, points_2d, axis, colors_box[k])
        # Draw 2D bbox (ground truth, obtained from instance masks)
        color_gt = (237, 63, 234)
        bboxes_gt = bboxes_gt[0]
        for bbox_gt in bboxes_gt:
            y1, x1, y2, x2 = bbox_gt
            # points_2d = np.array([[y1,x1], [y1,x2], [y2,x1], [y2,x2]])
            points_2d = np.array([[x1,y1], [x1,y2], [x2,y1], [x2,y2]])
            draw_bboxes_2d(im, points_2d, color_gt)
            
            
        box_save_path = str(output_path / f"{output_prefix}_bbox3d.png")
        print("imwrite")
        cv2.imwrite(box_save_path, np.copy(im))
        # dump_paths["pred_bboxes"] = box_save_path
        
        n += 1
        if n>=5: break
        
    
    

if __name__ == "__main__":
    args_list = None
    if len(sys.argv) > 1:
        args_list = sys.argv[1:]
    hparams = get_scene_grasp_model_params(args_list)
    main(hparams)