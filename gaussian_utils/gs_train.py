import numpy as np
import torch
from torch import Tensor, optim
import os
import sys
import time
from PIL import Image
from argparse import ArgumentParser, Namespace
from .arguments import ModelParams, PipelineParams, OptimizationParams
from .scene import GaussianModel

from .gs_renderer import render

def training(
        opt,
        gaussians: GaussianModel,
        w2c: np.ndarray,
        gt_image: Tensor,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        device = torch.device("cuda:0"),
    ):
    print(gt_image.shape)
    gt_image = gt_image.to(device)
    # optimizer = optim.Adam(
    #     [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
    # )
    gaussians.training_setup(opt)
    mse_loss = torch.nn.MSELoss()
    frames = []
    times = [0] * 2  # rasterization, backward
    
    for iter in range(iterations):
        # Render
        start = time.time()
        render_pkg = render(gaussians, w2c=w2c, render_mode="RGB")
        out_img = render_pkg["render_colors"][0]
        torch.cuda.synchronize()
        times[0] += time.time() - start
        
        # Loss, Optimizer step
        loss = mse_loss(out_img, gt_image)
        gaussians.optimizer.zero_grad()
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        times[1] += time.time() - start
        gaussians.optimizer.step()
        print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

        if save_imgs and iter % 20 == 0:
            frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
    if save_imgs:
        # save them as a gif with PIL
        frames = [Image.fromarray(frame) for frame in frames]
        out_dir = os.path.join(os.getcwd(), "gaussian_utils", "renders")
        os.makedirs(out_dir, exist_ok=True)
        frames[0].save(
            f"{out_dir}/training.gif",
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=5,
            loop=0,
        )
        # Initial rendering results
        frames[0].save(f"{out_dir}/initial.png")
        # # Save gt_image
        gt_image = (gt_image.detach().cpu().numpy() * 255).astype(np.uint8)
        gt_rgb = gt_image[:,:,:3]
        pil_image = Image.fromarray(gt_rgb)
        out_dir = os.path.join(os.getcwd(), "gaussian_utils", "renders")
        pil_image.save(f"{out_dir}/gt.png")

    print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
    print(
        f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
    )
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(model_path, retrieved_gsplat_fname))
            
    # # Initialize system state (RNG)
    # safe_state(args.quiet)

    # # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(op.extract(args), gaussians, gt_image)

    # All done
    print("\nTraining complete.")