import json
import numpy as np
import torch
import pathlib
import tqdm

from llava.train.cam_instance_intersect import (load_filemap, 
                                                calculate_best_fit_camera_extrinsics, 
                                                calculate_random_camera_extrinsics_cut_top_bottom,
                                                calculate_random_camera_extrinsics)



if __name__ == "__main__":
            
    scannet_data = json.load(open("playground/data/train_info/scanrefer_train_3d_llava.json", "r"))

    all_intersected_view_counts = []

    for source in tqdm.tqdm(scannet_data):
        scene_id = source['scene_id']
        scene_name = scene_id
        scan_folder = "playground/data/scannet"
        scan_data_path  = pathlib.Path(scan_folder) / 'train' / f'{scene_id}.pth'
        superpoint_path = pathlib.Path(scan_folder) / 'super_points' / f'{scene_id}.bin'


        assert "object_id" in source
        object_id = source['object_id']
        object_id = np.array(object_id).reshape(-1)


        filemap = load_filemap(scene_name=scene_name)

        object_indexes = np.where(filemap["valid_object_ids"] == object_id)[0]
        assert len(object_indexes) != 0, f"scene {scene_name} does not have object id {object_id}"
        object_index = object_indexes[0]
        # object_index = object_id + 1 # object ids has from -1, after sorting this -1 comes to index 0, so object_id + 1 gives the correct index
        intersections_column = filemap["cam_object_intersections"][:, object_index].reshape(-1)

        zero_mask = intersections_column == 0.

        all_intersected_view_counts.append(np.sum(~zero_mask))
    
    print(f"average number of selectable views for Scan2Cap is {np.array(all_intersected_view_counts).mean()}")