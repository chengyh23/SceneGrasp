#
# Oct 2024
# Johan C
# 
# cf. gaussian-splatting/gaussian_renderer/__init__.py
# 
import numpy as np
import torch
from .scene import GaussianModel
from gsplat import rasterization

def render(pc: GaussianModel, w2c: np.ndarray, render_mode):
    """ 
    Using rasaterization in https://docs.gsplat.studio/main/apis/rasterization.html
    """
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    opacity = torch.reshape(opacity, (-1,))
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    # # get extrinsic parameters (world-to-camera transform)
    # c2w = np.array(
    #     [
    #         [
    #             1.0,
    #             0.0,
    #             0.0,
    #             0.0
    #         ],
    #         [
    #             0.0,
    #             0.4717152714729309,
    #             -0.8817509412765503,
    #             -1.2469841241836548
    #         ],
    #         [
    #             0.0,
    #             0.8817508816719055,
    #             0.4717152714729309,
    #             0.6671061515808105
    #         ],
    #         [
    #             0.0,
    #             0.0,
    #             0.0,
    #             1.0
    #         ]
    #     ],
    # )
    # c2w[:3, 1:3] *= -1
    # w2c = np.linalg.inv(c2w)
    viewmat = torch.tensor(w2c, dtype=torch.float32, device=torch.device("cuda:0"),)
    # intrinsic parameters
    camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    intrinsics = camera_intrinsics
    K = torch.tensor(intrinsics, dtype=torch.float32, device=torch.device("cuda:0"))
    W, H = 640, 480
    
    render_colors, render_alphas, meta = rasterization(
        means3D,
        rotations,
        scales,
        opacity,
        shs,
        viewmat[None],
        K[None],
        W,
        H,
        packed=False,
        render_mode=render_mode,
        sh_degree=3,
    )
    # out_img = renders[0]
    # return out_img
    return {"render_colors": render_colors,
            "render_alphas": render_alphas,
            "meta": meta}
