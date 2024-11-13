import sys
CORSAIR_DIR = "/home/chengyh23/Documents/CORSAIR"
sys.path.append(CORSAIR_DIR)
from utils.visualize import Wvisualize  #CORSAIR

import os
import open3d as o3d

from scripts.rgbd2pcls import rgbd2pcls_object, rgbd2pcls_scene


"""
Ref. gsplat/examples/image_fitting.py
https://github.com/nerfstudio-project/gsplat
"""
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization
from plyfile import PlyData


class Gaussians:
    def __init__(self, ply_fname):
        """3DGS parameterizes the scene space with a set of Gaussian primitives, stacking the parameters together.
        Gaussian features/parameters: [C,O,S,R,SH] with centroids, opacities, scales and rotation, spherical harmonics.
        Args:
            plyply_fname_path (str): ShapeSplat model
        """
        root = "/media/sdb2/chengyh23/ShapeSplatsV1/gs_data/"
        ply_path = os.path.join(root, ply_fname)
        gs_vertex = PlyData.read(ply_path)['vertex']
        ### load centroids[x,y,z] - Gaussian centroid
        x = gs_vertex['x'].astype(np.float32)
        y = gs_vertex['y'].astype(np.float32)
        z = gs_vertex['z'].astype(np.float32)
        centroids = np.stack((x, y, z), axis=-1) # [n, 3]

        ### load o - opacity
        opacity = gs_vertex['opacity'].astype(np.float32).reshape(-1, 1)


        ### load scales[sx, sy, sz] - Scale
        scale_names = [
            p.name
            for p in gs_vertex.properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((centroids.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = gs_vertex[attr_name].astype(np.float32)

        ### load rotation rots[q_0, q_1, q_2, q_3] - Rotation
        rot_names = [
            p.name for p in gs_vertex.properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((centroids.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = gs_vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)

        ### load base sh_base[dc_0, dc_1, dc_2] - Spherical harmonic
        sh_base = np.zeros((centroids.shape[0], 3, 1))
        sh_base[:, 0, 0] = gs_vertex['f_dc_0'].astype(np.float32)
        sh_base[:, 1, 0] = gs_vertex['f_dc_1'].astype(np.float32)
        sh_base[:, 2, 0] = gs_vertex['f_dc_2'].astype(np.float32)
        sh_base = sh_base.reshape(-1, 3)
        
        self.device = torch.device("cuda:0")
        self.means = centroids
        self.scales = scales
        self.quats = rots
        self.opacities = opacity.squeeze()
        self.SH = sh_base
    
    def render(self, render_mode):
        self.means = torch.tensor(self.means, dtype=torch.float32, device=self.device)
        self.scales = torch.tensor(self.scales, dtype=torch.float32, device=self.device)
        self.quats = torch.tensor(self.quats, dtype=torch.float32, device=self.device)
        self.opacities = torch.tensor(self.opacities, dtype=torch.float32, device=self.device)
        self.SH = torch.tensor(self.SH, dtype=torch.float32, device=self.device)
        
        # viewmat = torch.tensor(
        #     [
        #         [1.0, 0.0, 0.0, -35.0],
        #         [0.0, 1.0, 0.0, 0.0],
        #         [0.0, 0.0, 1.0, 0.0],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ],
        #     device=self.device,
        # )
        viewmat = torch.tensor(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.4717152714729309,
                    -0.8817509412765503,
                    -1.2469841241836548
                ],
                [
                    0.0,
                    0.8817508816719055,
                    0.4717152714729309,
                    0.6671061515808105
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            device=self.device,
        )
        # K = self.get_intrinsic()
        fov_x = math.pi / 2.0
        W, H = 300, 300
        # self.focal = 0.5 * float(W) / math.tan(0.5 * fov_x)
        self.focal = 1.75
        
        intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
        K = torch.tensor(intrinsics, dtype=torch.float32, device=self.device)
        
        # K = torch.tensor(
        #     [
        #         [self.focal, 0, W / 2],
        #         [0, self.focal, H / 2],
        #         [0, 0, 1],
        #     ],
        #     device=self.device,
        # )
        renders, _, _ = rasterization(
            self.means,
            self.quats / self.quats.norm(dim=-1, keepdim=True),
            self.scales,
            self.opacities,
            self.SH,
            viewmat[None],
            K[None],
            300, # self.W,
            300, # self.H,
            packed=False,
            render_mode=render_mode,
        )
        out_img = renders[0]
        return out_img
        
class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
        gaussians: Gaussians = None # ShapeSplat
    ):
        """
        Args:
            gt_image (Tensor): [480, 640, 3]
        """
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.render_mode = "RGB"
        if gt_image.shape[2]==4:
            self.render_mode = "RGB+D"
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self.gaussians = gaussians
        # if gaussians is None:
        #     self._init_gaussians_random()
        # else:
        #     self.init_gaussians(gaussians)
            
    def init_gaussians(self, gaussians: Gaussians):
        """_summary_
        means: N,3
        scales: N,3
        opacities: N
        quats: N,4
        colors: 
        """
        self.means = torch.tensor(gaussians.means, dtype=torch.float32, device=self.device)
        self.scales = torch.tensor(gaussians.scales, dtype=torch.float32, device=self.device)
        d = 3
        self.rgbs = torch.tensor(gaussians.SH, dtype=torch.float32, device=self.device)
        self.quats = torch.tensor(gaussians.quats, dtype=torch.float32, device=self.device)
        self.opacities = torch.tensor(gaussians.opacities, dtype=torch.float32, device=self.device)
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)
        
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False
        
    def _init_gaussians_random(self):
        """Random gaussians"""
        bd = 2

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    
        
    def get_intrinsic(self):
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )

        # camera_k.gif
        demo_data_path = Path("outreach/demo_data")
        camera_k = np.loadtxt(demo_data_path / "camera_k.txt", dtype=np.float32)
        K = torch.tensor(camera_k[:3,:3], device=self.device)
        
        # # focal_x4.gif
        # K = torch.tensor(
        #     [
        #         [self.focal*4, 0, self.W / 2],
        #         [0, self.focal*4, self.H / 2],
        #         [0, 0, 1],
        #     ],
        #     device=self.device,
        # )
        return K

def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
    save_imgs: bool = False,
    img_path: Optional[Path] = None,
    iterations: int = 100,
    lr: float = 0.01,
) -> None:
    # Extract object pointcloud from the scene
    # root = "data/NOCSDataset/CAMERA/"
    # img_path = "val/00000/0002"
    # img_full_path = os.path.join(root, img_path)
    img_full_path = str(img_path)
    img_full_path = img_full_path[:img_full_path.find("_color.png")]
    # pcd = rgbd2pcls_object(img_full_path)
    pcd = rgbd2pcls_scene(img_full_path)
    aabbox_pcd = pcd.get_axis_aligned_bounding_box()

    # o3d.visualization.webrtc_server.enable_webrtc()
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    # o3d.visualization.draw([pcd, aabbox_pcd, coordinate_frame])
    # # Wvisualize([pcd, aabbox_pcd],["GREEN","BLACK"])
    # # print("got it")

    # Fit single image's gsplat
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )
    print("debug breakpoint")
    
    # means, scales, quats, rgbs, opacities = extract_object(trainer, aabbox_pcd)
    means = trainer.means.detach().cpu()
    # viewmat = trainer.viewmat.detach().cpu()
    # means[:, 2] += viewmat[2,3]
    rgbs = trainer.rgbs.detach().cpu()
    Wvisualize([means, np.asarray(pcd.points)],[rgbs, "BLACK"])
    

def extract_object(trainer, aabbox_pcd):
    means = trainer.means.detach().cpu()
    viewmat = trainer.viewmat.detach().cpu()
    means[:, 2] += viewmat[2,3]
    scales = trainer.scales.detach().cpu()
    quats = trainer.quats.detach().cpu()
    rgbs = trainer.rgbs.detach().cpu()
    opacities = trainer.opacities.detach().cpu()
    # means = trainer.means.detach().cpu().numpy()
    # scales = trainer.scales.detach().cpu().numpy()
    # quats = trainer.quats.detach().cpu().numpy()
    # rgbs = trainer.rgbs.detach().cpu().numpy()
    # opacities = trainer.opacities.detach().cpu().numpy()
    
    max_bound = torch.from_numpy(aabbox_pcd.max_bound)
    min_bound = torch.from_numpy(aabbox_pcd.min_bound)
    mask_object = torch.all((means <= max_bound) & (means >= min_bound), axis=1)
    if not torch.any(mask_object):
        print(f"no points within {aabbox_pcd.max_bound} and {aabbox_pcd.min_bound}")
    means_sub = means[mask_object]
    scales_sub = scales[mask_object]
    quats_sub = quats[mask_object]
    rgbs_sub = rgbs[mask_object]
    opacities_sub = opacities[mask_object]
    
    
    K = torch.tensor(
        [
            [trainer.focal, 0, trainer.W / 2],
            [0, trainer.focal, trainer.H / 2],
            [0, 0, 1],
        ],
        device=trainer.device,
    )
    renders, _, _ = rasterization(
        means_sub,
        quats_sub / quats_sub.norm(dim=-1, keepdim=True),
        scales_sub,
        opacities_sub,  # torch.sigmoid(self.opacities),
        rgbs_sub,   # torch.sigmoid(self.rgbs),
        trainer.viewmat[None],
        K[None],
        trainer.W,
        trainer.H,
        packed=False,
    )
    out_img = renders[0]
    img = Image.fromarray(out_img)
    img.save("object.png")
    print("Saved to object.png")
    return 0

if __name__ == "__main__":
    # tyro.cli(main)
    
    # # 1. Read o3d pcd from ShapeNet
    # model_path = "/media/sdb2/chengyh23/ShapeNetCore.v2.PC15k/02691156/val/969455251a1ee3061c517f0fe59ec7ee.npy"
    # model_npy = np.load(model_path)
    # from common.utils.misc_utils import get_o3d_pcd_from_np
    # # pcd = get_o3d_pcd_from_np(model_npy)
    
    # 2. Read o3d pcd from ShapeSplat
    ply_fpath="/media/sdb2/chengyh23/ShapeSplatsV1/gs_data/03642806-10f18b49ae496b0109eaabd919821b8.ply"
    # pcd = o3d.io.read_point_cloud(ply_fpath)
    # if pcd.is_empty():
    #     print("Failed to load point cloud. Please check the file path and format.")
    # else:
    #     print(f"Point cloud loaded successfully with {len(pcd.points)} points.")

    gaussians = Gaussians(ply_fpath)
    out_img = gaussians.render(render_mode='RGB')
    # out_img = out_img[:,:,-1]
    frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
    
    img = Image.fromarray(frame)
    img.save("object.png")
    print("Saved to object.png")
    
    # # Visualize
    # # o3d.visualization.webrtc_server.enable_webrtc()
    # # Wvisualize([model_npy],["GREEN"])
    # # o3d.visualization.draw_geometries([pcd], zoom=0.3412,
    # #                               front=[0.4257, -0.2125, -0.8795],
    # #                               lookat=[2.6172, 2.0475, 1.532],
    # #                               up=[-0.0694, -0.9768, 0.2024])
    # import open3d.visualization as vis
    # # o3d.visualization.draw(pcd,bg_color=(1.0, 1.0, 1.0, 1.0))
    # vis.draw([pcd],bg_color=[1,1,1])
    # # o3d.visualization.draw([pcd])
    
    # # vis = o3d.visualization.Visualizer()
    # # vis.create_window()
    # # vis.add_geometry(pcd)
    # # # Set background color (change to white or any color you prefer)
    # # vis.get_render_option().background_color = [1, 1, 1]  # RGB for white
    # # vis.run()
    # # vis.destroy_window()
