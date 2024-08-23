import torch
import argparse
import os
import sys
CORSAIR_DIR = "/home/chengyh23/Documents/CORSAIR"
sys.path.append(CORSAIR_DIR)
sys.path.append(CORSAIR_DIR+"/src")

from datasets.CategoryDataset import CategoryDataset
# from ...CORSAIR.src.corsair.datasets.shapenet.dataset_pcd import ShapeNetPointCloudDataset
from src.corsair.datasets.shapenet.dataset_pcd import ShapeNetPointCloudDataset
# from ...CORSAIR.src.config import get_config
from src.config import get_config
from common.config import config_dataset_details
from scene_grasp.scale_shape_grasp_ae.model.auto_encoder_scale_grasp import (
    PointCloudScaleBasedGraspAE,
)
def test_net(args, config):
    # Load the model
    estimator = PointCloudScaleBasedGraspAE(
        args.emb_dim, args.num_point, args.choose_bd_sign
    )
    estimator.cuda()
    estimator.load_state_dict(torch.load(args.model_path))

    config.catid = "03001627"
    config.root = "/media/sdb2/chengyh23/ShapeNetCore.v2.PCD.npy/sample15000/ncc/"
    # # dataset - CategoryDataset
    # dist_mat_root = "/media/sdb2/chengyh23/k8s/ShapeNetCore.v2.PCD.chamfer_distance/sample15000/ncc/dual_direction/"
    # test_dataset = CategoryDataset(config.root, "test", config.catid, dist_mat_root, 
    #                             config.pos_ratio, config.neg_ratio, config.voxel_size)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
    #                                         num_workers=2, collate_fn=test_dataset.collate_pair_fn)
    
    # dataset - ShapeNetPointCloudDataset
    config.split = "test"
    pcd_dataset_config = ShapeNetPointCloudDataset.Config(
        root=config.root,
        synset_ids=[config.catid],
        split=config.split,
    )
    test_dataset = ShapeNetPointCloudDataset(pcd_dataset_config)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                                            num_workers=2,)
    
    # test
    for batch_xyz in test_loader:
        # base_output = model(base_input)
        batch_scales = torch.ones((config.batch_size, 1))
        batch_xyz = batch_xyz.cuda().to(torch.float)
        batch_scales = batch_scales.cuda().to(torch.float)
        _, endpoints = estimator(batch_xyz, batch_scales)
        # get transformation mat from output
        # T_est = 
        t_loss, r_loss = eval_pose(T_est)
    
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--dataset_root",
    #     type=str,
    #     default=str(config_global_paths.PROJECT_DATA_ROOT / "nocs_grasp_dataset"),
    # )
    parser.add_argument(
        "--num_point",
        type=int,
        default=config_dataset_details.get_n_points(),
        help="number of points, needed if use points",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=config_dataset_details.get_emb_dim(),
        help="dimension of latent embedding",
    )
    parser.add_argument(
        "--choose_bd_sign",
        type=bool,
        default=False,
        help="""
        If passed, decide the sign of baseline-direction analytically based on
        the predicted point-cloud
        """,
    )
    parser.add_argument(
        "--model_path", 
        type=str,
        default="checkpoints/scale_ae.pth"
    )
    args = parser.parse_args()
    config = get_config()
    test_net(args, config)
    
if __name__ == "__main__":
    # main()
    data_dir = "data/NOCSDataset/"
    source = "CAMERA"
    img_path = "val/00008/0002"
    from common.utils.nocs_utils import load_depth, align_nocs_to_depth
    from scene_grasp.scene_grasp_net.data_generation.generate_data_nocs import process_data
    img_full_path = os.path.join(data_dir, source, img_path)
    depth_full_path_real = img_full_path + '_depth.png'
    all_exist = os.path.exists(img_full_path + '_color.png') and \
                os.path.exists(img_full_path + '_coord.png') and \
                os.path.exists(img_full_path + '_depth.png') and \
                os.path.exists(img_full_path + '_mask.png') and \
                os.path.exists(img_full_path + '_meta.txt') and \
                os.path.exists(depth_full_path_real)
    if not all_exist:
        print("not all exist")
        exit
    depth = load_depth(depth_full_path_real)
    masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
    