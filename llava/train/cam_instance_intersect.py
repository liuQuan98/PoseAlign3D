import numpy as np
import torch
import os



def load_filemap(scene_name, filepath_base="playground/data/view_instance_intersection"):
    return np.load(os.path.join(filepath_base, f"{scene_name}_view_instance_intersection_with_visibility.npz"), allow_pickle=True)


def calculate_best_fit_camera_extrinsics(scene_name, object_id, filemap, pose_data_path_base="playground/poses/scans", cut_ratio=None):
    """
    retrieve the array of camera-instance intersections from filemap and return the pose of the highest
    """
    if object_id == -1: # object not located, then randomly select a pose
        pose = None
        files = os.listdir(os.path.join(pose_data_path_base, scene_name, "pose"))
        while pose is None or np.isnan(pose).any() or np.isinf(pose).any():
            select = np.random.randint(0, len(files))
            pose = np.loadtxt(os.path.join(pose_data_path_base, scene_name, "pose", files[select]))
        return pose

    object_indexes = np.where(filemap["valid_object_ids"] == object_id)[0]
    assert len(object_indexes) != 0, f"scene {scene_name} does not have object id {object_id}"
    object_index = object_indexes[0]
    intersections_column = filemap["cam_object_intersections"][:, object_index].reshape(-1)
    pose_filenames = filemap["pose_files"]

    zero_mask = intersections_column == 0.

    intersections_column = intersections_column[~zero_mask]
    pose_filenames = np.array(pose_filenames)[~zero_mask]

    if len(intersections_column) == 0:
        select = np.random.randint(0, filemap["pose_files"].shape[0])
        return np.loadtxt(os.path.join(pose_data_path_base, scene_name, "pose", pose_filenames[select]))

    max_index = np.argmax(intersections_column)
    return np.loadtxt(os.path.join(pose_data_path_base, scene_name, "pose", pose_filenames[max_index]))


def calculate_random_camera_extrinsics_cut_top_bottom(scene_name, object_id, filemap, pose_data_path_base="playground/poses/scans", cut_ratio=0.2):
    """
    retrieve the array of camera-instance intersections from filemap and return the pose of the highest
    """
    if object_id == -1: # object not located, then randomly select a pose
        pose = None
        files = os.listdir(os.path.join(pose_data_path_base, scene_name, "pose"))
        while pose is None or np.isnan(pose).any() or np.isinf(pose).any():
            select = np.random.randint(0, len(files))
            pose = np.loadtxt(os.path.join(pose_data_path_base, scene_name, "pose", files[select]))
        return pose

    object_indexes = np.where(filemap["valid_object_ids"] == object_id)[0]
    assert len(object_indexes) != 0, f"scene {scene_name} does not have object id {object_id}"
    object_index = object_indexes[0]
    # object_index = object_id + 1 # object ids has from -1, after sorting this -1 comes to index 0, so object_id + 1 gives the correct index
    intersections_column = filemap["cam_object_intersections"][:, object_index].reshape(-1)
    pose_filenames = filemap["pose_files"]

    if len(filemap['pose_files']) == 0:
        return np.identity(4)

    # mask out the zero ones to speed up sorting
    zero_mask = intersections_column == 0.
    intersections_column = intersections_column[~zero_mask]
    pose_filenames = np.array(pose_filenames)[~zero_mask]

    if len(intersections_column) == 0:
        select = np.random.randint(0, filemap["pose_files"].shape[0])
        return np.loadtxt(os.path.join(pose_data_path_base, scene_name, "pose", filemap["pose_files"][select]))

    sorted_indices = np.argsort(intersections_column)
    assert cut_ratio < 0.5, "cut_ratio should be less than 0.5"
    cut_num = int(len(sorted_indices) * cut_ratio)

    valid_indices = np.array([])
    while len(valid_indices) == 0 and cut_num > 0:
        # select the middle indices after cutting top and bottom. If None left, shrink the cut_num by 1 and try again
        valid_indices = sorted_indices[cut_num:-cut_num] # this will work only when cut_num >= 1, since arr[0:-0] always returns an empty array
        cut_num -= 1

    if cut_num == 0 and len(sorted_indices) != 0:
        # deal with arr[0:-0] case, this only happens when len(intersection_column) is very small. In this case we just select from all candidates
        valid_indices = sorted_indices

    select = np.random.choice(valid_indices)
    max_index = sorted_indices[select]
    return np.loadtxt(os.path.join(pose_data_path_base, scene_name, "pose", pose_filenames[max_index]))


def calculate_random_camera_extrinsics(scene_name, object_id, filemap, pose_data_path_base="playground/poses/scans", cut_ratio=None):
    """
    retrieve the array of camera-instance intersections from filemap and return a random pose from them, as an ablation experiment
    """

    pose_filenames = np.array(filemap["pose_files"])
    select = np.random.randint(0, pose_filenames.shape[0])
    return np.loadtxt(os.path.join(pose_data_path_base, scene_name, "pose", pose_filenames[select]))
