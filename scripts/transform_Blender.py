#
# Write camera parameters into json,
# to use https://github.com/graphdeco-inria/gaussian-splatting
#
import os
import math
import json
import numpy as np


def camera_params2transform(intrinsics, extrinsics):
    """
    Write camera parameters into json,
    to use https://github.com/graphdeco-inria/gaussian-splatting
    """
    camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    # real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    
    
    fovx = 2 * math.atan(320 / 577.5)
    
    # Data to be written to JSON file
    data = {
        "camera_angle_x": fovx,
        "frames": [
            {
                "file_path": "NOCS/CAMERA/val/00000/0000",
                "rotation": 0.08726646259971647,
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.4717152714729309, -0.8817509412765503, -1.2469841241836548],
                    [0.0, 0.8817508816719055, 0.4717152714729309, 0.6671061515808105],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            }
        ]
    }

    file_path = "/media/sdb2/chengyh23/ShapeSplatsV1/render/transform_NOCS.json"
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"File written to {file_path}")

def check_poses_NOCS_train():
    """count number of matrices in NOCS pose.txt"""
    root_path = "/media/sdb2/chengyh23/NOCS/tmp/real_train"
    file_path = os.path.join(root_path, "scene_1", "pose.txt")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Each matrix has 4 rows, so divide total lines by 4 to get number of matrices
    num_matrices = len(lines) // 4
    print(num_matrices, len(lines))
    
if __name__ == "__main__":
    
    # camera_params2transform(None, None)
    check_poses_NOCS_train()