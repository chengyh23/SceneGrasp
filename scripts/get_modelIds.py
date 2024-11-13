import sys
from common.utils.scene_grasp_utils import (
    SceneGraspModel,
)
from common.utils.misc_utils import (
    get_scene_grasp_model_params,
)
from tqdm import tqdm
def get_modelIds(hparams, ):
    scene_grasp_model = SceneGraspModel(hparams)
    data_generator = scene_grasp_model.model.val_dataloader()
    all_modelIds = set()
    cnt = 0
    for batch in tqdm(data_generator):
        image, seg_target, modelIds, instance_ids, depth_target, pose_targets, bboxes_gt, _, img_name, scene_name = batch
        modelIds = modelIds[0]
        all_modelIds.update(modelIds)
        print(modelIds, all_modelIds)
        cnt += 1
        if cnt > 5: break
    return all_modelIds
        

if __name__ == "__main__":
    # Write stat
    args_list = None
    if len(sys.argv) > 1:
        args_list = sys.argv[1:]
    hparams = get_scene_grasp_model_params(args_list)
    get_modelIds(hparams)