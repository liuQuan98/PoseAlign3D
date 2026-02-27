import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import pathlib
import torch
from llava.train.cam_instance_intersect import load_filemap, calculate_best_fit_camera_extrinsics, calculate_random_camera_extrinsics_cut_top_bottom


def max_yaw_difference(vectors):
    """
    Compute the maximum yaw (azimuth) difference between any two 3D direction vectors.
    Args:
        vectors (np.ndarray): shape (N, 3), each row is a 3D direction vector.
    Returns:
        float: maximum yaw difference in radians,

    """
    # Normalize vectors (optional, just for safety)
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Compute yaw angles in [-pi, pi]
    yaws = np.arctan2(v[:, 1], v[:, 0])  # yaw = atan2(y, x)
    
    # Compute pairwise yaw differences
    diff = np.abs(yaws[:, None] - yaws[None, :])
    diff = np.minimum(diff, 2 * np.pi - diff)  # account for wrap-around
    
    # Return the maximum difference
    return np.max(diff), np.argmax(diff)


def unit_vector_from_to(v1, v2):
    """Compute the unit vector from point v1 to point v2."""
    vec = v2 - v1
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0])  # default unit vector if points are the same
    return vec / norm + 1e-8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_file_path", type=str, required=True, default="playground/data/train_info/scanqa_train_3d_llava.json")
    parser.add_argument("--pose_data_file_base", type=str, default="playground/poses/scans")
    parser.add_argument("--scannet_raw_scan_folder", type=str, default="playground/data/scannet")
    parser.add_argument("--view_instance_intersection_path", type=str, default="playground/data/view_instance_intersection")
    parser.add_argument(
        "--extra_data_file",
        type=str,
        default="playground/data/complementary_info/matched_ScanQA_v1.0_train.json",
        help="Path to the extra data file containing complementary information on scanqa.",
    )
    args = parser.parse_args()

    cut_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.49]

    data = json.load(open(args.training_file_path, "r"))
    with open(args.extra_data_file, 'r') as f:
        extra_data_dict = json.load(f)

    obj_view_differences = {}

    for ratio in cut_ratios:
        obj_view_differences[ratio] = {}

    for i, item in tqdm(enumerate(data), total=len(data)):
        scene_name = item["scene_id"]

        if 'scanqa' in args.training_file_path:
            object_ids = extra_data_dict[i]
        else:
            object_ids = item["object_id"]

        if type(object_ids) != list:
            object_ids = [object_ids]

        filemap = load_filemap(scene_name, filepath_base=args.view_instance_intersection_path)

        scan_data_path  = pathlib.Path(args.scannet_raw_scan_folder) / 'train' / f'{scene_name}.pth'
        raw_data = torch.load(scan_data_path)
        coord = raw_data['coord']
        instance = raw_data['instance_gt']

        poses = []
        for object_id in object_ids:
            pt_segment = np.isin(instance, object_id)
            object_center = coord[pt_segment].mean(0)

            object_indexes = np.where(filemap["valid_object_ids"] == object_id)[0]
            assert len(object_indexes) != 0, f"scene {scene_name} does not have object id {object_id}"
            object_index = object_indexes[0]
            intersections_column = filemap["cam_object_intersections"][:, object_index].reshape(-1)
            pose_filenames = filemap["pose_files"]

            if len(filemap['pose_files']) == 0:
                continue # no pose available, do not count as oppositr views

            # mask out the zero ones to speed up sorting
            zero_mask = intersections_column == 0.
            intersections_column = intersections_column[~zero_mask]
            pose_filenames = np.array(pose_filenames)[~zero_mask]

            if len(intersections_column) == 0:
                continue # no valid intersection, do not count as opposite views
 
            sorted_indices = np.argsort(intersections_column)

            for cut_ratio in cut_ratios:
                cut_num = int(len(sorted_indices) * cut_ratio)

                valid_indices = np.array([])
                while len(valid_indices) == 0 and cut_num > 0:
                    # select the middle indices after cutting top and bottom. If None left, shrink the cut_num by 1 and try again
                    valid_indices = sorted_indices[cut_num:-cut_num] # this will work only when cut_num >= 1, since arr[0:-0] always returns an empty array
                    cut_num -= 1

                if cut_num == 0 and len(sorted_indices) != 0:
                    # deal with arr[0:-0] case, this only happens when len(intersection_column) is very small. In this case we just select from all candidates
                    valid_indices = sorted_indices

                selected_poses = [np.loadtxt(os.path.join(args.pose_data_file_base, scene_name, "pose", pose_filenames[idx])) for idx in valid_indices]
                views_vectors = [unit_vector_from_to(item[:3, 3], object_center) for item in selected_poses]  # assuming the camera forward direction is the third column of the rotation matrix
                assert len(views_vectors) > 0, "No valid views found after cutting."
                views_vectors = np.array(views_vectors)
                max_diff, max_diff_position = max_yaw_difference(views_vectors) # float, array of shape (n_views, n_views)

                max_diff_positions = np.unravel_index(max_diff_position, (len(views_vectors), len(views_vectors)))
                view1, view2 = selected_poses[max_diff_positions[0].item()], selected_poses[max_diff_positions[1].item()]

                key = f"index_{i:06d}_object_{object_id:03d}"
                obj_view_differences[cut_ratio][key] = {'diff': np.degrees(max_diff),
                                                        'scene_name': scene_name,
                                                        'index': i,
                                                        'object_id': object_id,
                                                        'view1': view1.tolist(),
                                                        'view2': view2.tolist()}
                if np.degrees(max_diff) > 150:
                    print(f"Large yaw difference found: {np.degrees(max_diff)} degrees for {key}, cut_ratio={cut_ratio}")

    output_file = args.training_file_path.replace(".json", "_opposite_yaw_differences.json")
    with open(output_file, "w") as f:
        json.dump(obj_view_differences, f, indent=4)
    print(f"Saved opposite yaw differences to {output_file}")

