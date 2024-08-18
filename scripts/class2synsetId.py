import os
from tqdm import tqdm

def check_meta(img_path):
    """ Ref. process_data in scene_grasp/scene_grasp_net/data_generation/generate_data_nocs.py"""
    cls2synset_dict = {}
    meta_path = img_path + '_meta.txt'
    with open(meta_path, 'r') as f:
        i = 0
        for line in f:
            line_info = line.strip().split(' ')
            inst_id = int(line_info[0])
            cls_id = int(line_info[1])
            # # background objects and non-existing objects
            # if cls_id == 0 or (inst_id not in all_inst_ids):
            #     continue
            if len(line_info) == 3:
                model_id = line_info[2]    # Real scanned objs
            else:
                synset_id = line_info[2]
                model_id = line_info[3]    # CAMERA objs
            if cls_id not in cls2synset_dict:
                cls2synset_dict[cls_id] = {synset_id}
            else:
                cls2synset_dict[cls_id].add(synset_id)
            # remove one mug instance in CAMERA train due to improper model
            if model_id == 'b9be7cfe653740eb7633a2dd89cec754' or model_id == 'd3b53f56b4a7b3b3c9f016d57db96408':
                continue
    return cls2synset_dict

CLS2SYNSET_ID = {
    1: "02876657",
    2: "02880940",
    3: "02942699",
    4: "02946921",
    5: "03642806",
    6: "03797390",
}
WORD2CLS_ID = {
    "bottle":   1,
    "bowl":     2,
    "camera":   3,
    "can":      4,
    "laptop":   5,
    "mug":      6,
}
WORD2SYNSET_ID = {
    "bottle":   "02876657",
    "bowl":     "02880940",
    "camera":   "02942699",
    "can":      "02946921",
    "laptop":   "03642806",
    "mug":      "03797390",
}
if __name__ == "__main__":
    """
    Iterate images' meta info, Check the mapping from class_id to synsetId.
    Ref. annotate_test_data in scene_grasp/scene_grasp_net/data_generation/generate_data_nocs.py
    ----------------
    cls synset word
    1 02876657 bottle
    2 02880940 bowl
    3 02942699 camera
    4 02946921 can
    5 03642806 laptop
    6 03797390 mug
    ----------------
    """
    data_dir = "data/NOCSDataset"
    source = 'CAMERA'
    
    cls2synsetId = {}
    for class_id in range(7):
        cls2synsetId[class_id] = set()
    camera_val = open(os.path.join(data_dir, 'CAMERA', 'val_list_all.txt')).read().splitlines()
    img_list = camera_val
    for img_ind, img_path in tqdm(enumerate(img_list),desc="Processing images"):

        img_full_path = os.path.join(data_dir, source, img_path)
        _cls2synset = check_meta(img_full_path)
        for key,val in _cls2synset.items():
            cls2synsetId[key].update(val)
    
    for class_id, synset_ids in cls2synsetId.items():
        print(class_id, synset_ids)
        