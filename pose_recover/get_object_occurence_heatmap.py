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
    
    dataset = "scanrefer"
    cut_ratio = 0.49
    data = json.load(open(f"playground/data/train_info/{dataset}_train_3d_llava.json", "r"))

    all_intersected_view_counts = []

    world_coordinates = []
    ego_coordinates = []

    for source in tqdm.tqdm(data):
        scene_id = source['scene_id']
        scene_name = scene_id
        scan_folder = "playground/data/scannet"
        scan_data_path  = pathlib.Path(scan_folder) / 'train' / f'{scene_id}.pth'
        superpoint_path = pathlib.Path(scan_folder) / 'super_points' / f'{scene_id}.bin'


        raw_data = torch.load(scan_data_path)
        coord = raw_data['coord']
        color = raw_data['color']
        segment = raw_data['semantic_gt20']
        instance = raw_data['instance_gt']


        assert "object_id" in source
        object_id = source['object_id']
        object_id = np.array(object_id).reshape(-1)
        object_id_tmp = np.random.choice(object_id.reshape(-1))

        obj_mask = instance == object_id_tmp
        coord_object = coord[obj_mask]
        obj_center = np.mean(coord_object, axis=0)
        obj_cenetr_homogeneous = np.ones(4)
        obj_cenetr_homogeneous[:3] = obj_center

        scene_mean = np.mean(coord, axis=0)
        world_frame_coords = obj_center - scene_mean

        filemap = load_filemap(scene_name=scene_name)

        possible_camera_extrinsic = calculate_random_camera_extrinsics_cut_top_bottom(
            scene_name=scene_name, object_id=object_id_tmp, filemap=filemap,
            pose_data_path_base="playground/poses/scans", cut_ratio=cut_ratio
        )

        camera_frame_coords = np.linalg.inv(possible_camera_extrinsic) @ obj_cenetr_homogeneous
        shift_from_rfd_to_flu = np.array([[0, 0, 1, 0],
                                         [-1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [0, 0, 0, 1]])
        camera_frame_coords_flu = shift_from_rfd_to_flu @ camera_frame_coords

        # print(world_frame_coords, camera_frame_coords_flu, flush=True)
        world_coordinates.append(world_frame_coords)
        ego_coordinates.append(camera_frame_coords_flu)

    np.savez(f"object_heatmap_{dataset}_cut{cut_ratio}", world_coords=world_coordinates, ego_coords=ego_coordinates)