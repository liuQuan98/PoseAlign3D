import numpy as np
import os
import glob
import torch
import tqdm


def load_scan_data(scan_data_path):
    raw_data = torch.load(scan_data_path)
    coord = raw_data['coord']
    color = raw_data['color']
    segment = raw_data['semantic_gt20']
    instance = raw_data['instance_gt']

    return {
        'coord': coord,
        'color': color,
        'segment': segment,
        'instance': instance,
    }


def build_depth_buffer(scene_points, fx, fy, cx, cy, width, height, T_cw):
    """Build a Z-buffer (depth map) by projecting scene points into the camera."""
    Pw = np.hstack([scene_points, np.ones((len(scene_points), 1))])
    Pc = (T_cw @ Pw.T).T
    Xc, Yc, Zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]

    valid = Zc > 0
    u = (fx * Xc / Zc + cx).astype(int)
    v = (fy * Yc / Zc + cy).astype(int)

    mask = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    depth = np.full((height, width), np.inf)
    for uu, vv, zz in zip(u[mask], v[mask], Zc[mask]):
        if zz < depth[vv, uu]:
            depth[vv, uu] = zz
    return depth


def visible_fraction(depth, target_points, fx, fy, cx, cy, width, height, T_cw):
    """
    Compute the fraction of target points that are visible from a given camera pose,
    using a precomputed Z-buffer for occlusion testing.

    Returns
    -------
    pixel_ratio : float
        Fraction of in-frustum points that are visible (not occluded).
    point_ratio : float
        Fraction of all target points that are visible.
    """
    Pw = np.hstack([target_points, np.ones((len(target_points), 1))])
    Pc = (T_cw @ Pw.T).T
    Xc, Yc, Zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]

    u = (fx * Xc / Zc + cx).astype(int)
    v = (fy * Yc / Zc + cy).astype(int)

    mask = (Zc > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    hit = Zc[mask] <= depth[v[mask], u[mask]] + 1e-3

    pixel_ratio = hit.sum() / len(Zc[mask]) if len(Zc[mask]) > 0 else 0.0
    point_ratio = hit.sum() / len(Zc) if len(Zc) > 0 else 0.0

    return pixel_ratio, point_ratio


if __name__ == "__main__":
    # Camera intrinsics for ScanNet
    fx, fy, cx, cy = 1169.621094, 1167.105103, 646.295044, 489.927032
    width, height = 1296, 968

    pose_sample_interval = 25  # 25 means only 4% of the poses are left. This is reasonable as the original ScanNet poses are very dense.

    # Shrink the image for larger point collision region (larger angle per pixel)
    image_shrinkage = 0.1
    fx, fy, cx, cy = fx * image_shrinkage, fy * image_shrinkage, cx * image_shrinkage, cy * image_shrinkage
    width, height = int(width * image_shrinkage), int(height * image_shrinkage)

    base_path = "./playground"
    for split in ['train', 'val']:
        merged_data_file_path_base = os.path.join(base_path, "data", "scannet", split)
        poses_file_path = os.path.join(base_path, "poses/scans")

        merged_data_files = glob.glob(os.path.join(merged_data_file_path_base, "*.pth"))
        data_file_keys = [item[-16:-4] for item in merged_data_files]

        output_path = os.path.join(base_path, 'data', 'view_instance_intersection')
        os.makedirs(output_path, exist_ok=True)

        for data_file_key in tqdm.tqdm(data_file_keys):
            posefiles = sorted(os.listdir(os.path.join(poses_file_path, data_file_key, 'pose')),
                               key=lambda x: int(x.split('.')[0]))
            scene_data_path = os.path.join(merged_data_file_path_base, f"{data_file_key}.pth")
            raw_data = load_scan_data(scene_data_path)
            valid_object_ids = sorted(np.unique(raw_data['instance']))

            # Filter out pose files with NaN or Inf values
            filtered_posefiles = []
            for posefile in posefiles:
                pose_tmp = np.loadtxt(os.path.join(poses_file_path, data_file_key, 'pose', posefile))
                if not (np.isnan(pose_tmp).any() or np.isinf(pose_tmp).any()):
                    filtered_posefiles.append(posefile)

            # Subsample poses to reduce computation
            if pose_sample_interval > 1:
                filtered_posefiles = filtered_posefiles[::pose_sample_interval]

            cam_object_intersections = np.zeros((len(filtered_posefiles), len(valid_object_ids)), dtype=np.float32)

            for i, posefile in enumerate(filtered_posefiles):
                pose = np.loadtxt(os.path.join(poses_file_path, data_file_key, 'pose', posefile))

                # Precompute the depth buffer for the whole scene, reused for all objects
                z_buffer = build_depth_buffer(raw_data['coord'], fx, fy, cx, cy, width, height, np.linalg.inv(pose))

                try:
                    for j, object_id in enumerate(valid_object_ids):
                        pt_segment = np.isin(raw_data['instance'], object_id)
                        assert pt_segment.sum() > 0, "Object instance ID not found in the scene."
                        inlier_coord = raw_data['coord'][pt_segment]

                        # Compute visible fraction by comparing projected object points with the depth buffer
                        _, cam_object_intersections[i, j] = visible_fraction(z_buffer, inlier_coord, fx, fy, cx, cy, width, height, np.linalg.inv(pose))
                except Exception as e:
                    print("object_id:", object_id, "scene:", scene_data_path,
                          "posefile:", os.path.join(poses_file_path, data_file_key, 'pose', posefile),
                          "\nerror:", e)
                    raise ValueError("Error in computing frustum intersection.")

            np.savez(os.path.join(output_path, f"{data_file_key}_view_instance_intersection_with_visibility.npz"),
                     cam_object_intersections=cam_object_intersections,
                     pose_files=filtered_posefiles,
                     valid_object_ids=valid_object_ids)
        print(f"Saved {split} view-instance intersection data to {output_path}/", flush=True)
