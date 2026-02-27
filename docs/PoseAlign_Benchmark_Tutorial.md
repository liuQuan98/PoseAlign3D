# Applying PoseAlign to New 3D Benchmarks

This tutorial explains how to apply the PoseRecover + PoseAlign pipeline to an **arbitrary 3D point cloud dataset** beyond ScanNet. For reproducing our paper results on ScanNet, see the main [README](../README.md) instead.

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Prerequisites](#2-prerequisites)
- [3. Stage 1: PoseRecover — Computing View-Instance Intersection Maps](#3-stage-1-poserecover--computing-view-instance-intersection-maps)
- [4. Stage 2: Pose Selection at Training and Inference Time](#4-stage-2-pose-selection-at-training-and-inference-time)
- [5. Stage 3: PoseAlign — Point Cloud Transformation](#5-stage-3-posealign--point-cloud-transformation)
- [6. Adapting Training](#6-adapting-training)
- [7. Adapting Evaluation](#7-adapting-evaluation)
- [8. Coordinate System Reference](#8-coordinate-system-reference)
- [9. Porting Checklist](#9-porting-checklist)

---

## 1. Introduction

Existing 3D point cloud benchmarks (ScanQA, ScanRefer, Scan2Cap, etc.) contain directional queries — "What is to the **left** of the table?" — but provide no ego pose to ground directions. PoseAlign solves this with a two-stage pipeline:

1. **PoseRecover** (offline, run once): For each scene, compute a visibility matrix between every camera pose and every object instance. At training/inference time, use this matrix to select the best-matching ego pose for each question.

2. **PoseAlign** (applied at training/inference): Transform the point cloud (or the model's representation) so that it reflects the selected ego pose. Three pathways are available:
   - **PoseAlign-Transform**: rotate the point cloud into the camera's egocentric frame
   - **PoseAlign-Prompt**: inject a textual pose description into the language prompt
   - **PoseAlign-Embed**: concatenate per-point pose features before the multimodal projector

This tutorial walks through each stage and explains what to adapt for a new dataset.

---

## 2. Prerequisites

Your dataset must provide:

| Requirement | What it is | Example |
|---|---|---|
| **Point cloud scenes** | Per-scene 3D coordinates | `coord` (N, 3) |
| **Camera poses** | 4x4 extrinsic matrices from an RGB-D video sequence | One `.txt` file per frame for ScanNet. If extrinsics are lost, you can try generating artificial ones with SLAM |
| **Camera intrinsics** | Focal lengths and principal point | `fx, fy, cx, cy, width, height` |
| **Object-level annotations** | QA/captioning data that references objects by instance ID which could be linked to certain annotation files | `{"question": "...", "object_id": 42, "scene_id": "scene0001"}` |
| **Label of your choice** | One among three kinds of labels: (1) instance segementation mask, (2) object detection bboxes, (3) object center point cooridnate (xyz) | `instance_gt` (N,) |

**Not required**: color image, depth image, or mesh data.

<!-- > **Note on annotations without object IDs**: If your benchmark's annotations do not include explicit object IDs (as is the case with 3D-LLaVA ScanQA dataset), we can try to recover a supplementary mapping from each sample to the relevant object ID(s) using the raw dataset. See [Section 6](#6-adapting-training) for details. -->

---

## 3. Stage 1: PoseRecover — Computing View-Instance Intersection Maps

### 3.1 Overview

PoseRecover computes, for every (camera, object) pair in a scene, a **visibility score**: the fraction of the object's points that are visible from that camera, accounting for occlusion.

The pipeline:
1. For each camera pose, build a **Z-buffer** (depth map) by projecting all scene points into the camera.
2. For each object, project its points into the same camera and check against the Z-buffer. A point is "visible" if its depth matches the buffer (within 1mm tolerance), meaning it is not occluded by closer geometry.
3. Store the resulting `(num_cameras, num_objects)` matrix as a compressed `.npz` file.

### 3.2 Adapting the Script

The reference implementation is [`pose_recover/export_pose_intersections_with_visibility_check.py`](../pose_recover/export_pose_intersections_with_visibility_check.py). Below is each section you need to modify.

#### Camera intrinsics (line 71)

Replace with your dataset's intrinsics:

```python
# Original (ScanNet):
fx, fy, cx, cy = 1169.621094, 1167.105103, 646.295044, 489.927032
width, height = 1296, 968

# Your dataset:
fx, fy, cx, cy = YOUR_FX, YOUR_FY, YOUR_CX, YOUR_CY
width, height = YOUR_WIDTH, YOUR_HEIGHT
```

#### Pose sampling interval (line 74)

ScanNet provides ~2,500 frames per scene at 30 fps. With `pose_sample_interval = 25`, only ~4% of frames are kept. Adjust based on your video density:

```python
# Dense video (>1000 frames/scene): subsample aggressively
pose_sample_interval = 25

# Sparse video (<200 frames/scene): keep all or nearly all
pose_sample_interval = 1
```

#### Image shrinkage factor (line 77)

This downscales the virtual image plane, making each pixel cover a larger angular region. This acts as a tolerance factor — a point that projects 5 pixels away from the nearest occupied pixel in the original resolution may land on the same pixel after shrinkage. A value of `0.1` (10x downscale) works well for ScanNet's resolution. For lower-resolution cameras, use a larger factor (e.g., `0.2`–`0.5`).

#### Data loading (function `load_scan_data`)

Adapt to your point cloud format. The function must return a dict with at minimum:

```python
{
    'coord': np.ndarray,    # (N, 3) point coordinates in world frame
    'instance': np.ndarray, # (N,) per-point instance IDs (integers)
}
```

#### Pose file loading (lines 92–104)

Adapt the directory structure and file format. The current code expects:

```
playground/poses/scans/<scene_id>/pose/<frame_number>.txt
```

where each `.txt` contains a 4x4 camera-to-world matrix. If your poses are stored differently (e.g., a single JSON per scene, or in a different matrix convention), modify the loading loop accordingly. See [Section 8](#8-coordinate-system-reference) for coordinate convention details.

#### Output format

The output `.npz` must contain these exact keys (consumed by `cam_instance_intersect.py`):

| Key | Shape | Description |
|---|---|---|
| `cam_object_intersections` | `(num_cameras, num_objects)` | Float32 visibility scores |
| `pose_files` | `(num_cameras,)` | Filenames of the retained camera poses |
| `valid_object_ids` | `(num_objects,)` | Sorted unique instance IDs in the scene |

### 3.3 Key Functions

**`build_depth_buffer(scene_points, fx, fy, cx, cy, width, height, T_cw)`**

Projects all scene points into the camera using the world-to-camera transform `T_cw = inv(T_wc)`. For each pixel, retains the closest depth. No changes needed if your dataset uses a standard pinhole camera model.

**`visible_fraction(depth, target_points, fx, fy, cx, cy, width, height, T_cw)`**

Projects a subset of points (one object) and checks each against the depth buffer. Returns:
- `pixel_ratio`: fraction of in-frustum object points that pass the occlusion test
- `point_ratio`: fraction of *all* object points that are visible (this is the value stored in the intersection matrix)

### 3.4 Verifying Output

After running PoseRecover, verify the output is sensible:

1. **Check coverage**: adapt `pose_recover/examine_scannet_intersect_metrics.py` to your dataset. It reports the average number of camera views with non-zero visibility per object. If this is 0 for most objects, there is likely an intrinsics or extrinsics convention mismatch.

2. **Spot-check individual scenes**: load an NPZ and inspect:
   ```python
   data = np.load("scene_xxx_view_instance_intersection_with_visibility.npz")
   scores = data["cam_object_intersections"]
   print(f"Shape: {scores.shape}")              # (num_cameras, num_objects)
   print(f"Non-zero entries: {(scores > 0).sum() / scores.size:.1%}")
   print(f"Max score: {scores.max():.3f}")       # should be < 1.0
   ```

<!-- 3. **Visualize**: use `pose_recover/get_object_occurence_heatmap.py` as a template to plot object positions in both world and egocentric frames, confirming the coordinate transform is correct. -->

---

## 4. Stage 2: Pose Selection at Training and Inference Time

### 4.1 Overview

At each training step (or inference sample), a pose is selected from the precomputed intersection matrix for the relevant object. The module `llava/train/cam_instance_intersect.py` provides three strategies:

| Strategy | Function | Training Flag | Description |
|---|---|---|---|
| **Top** | `calculate_best_fit_camera_extrinsics()` | *(default)* | Selects the camera with the highest visibility score. Deterministic; may overfit to a single viewpoint |
| **Clip** | `calculate_random_camera_extrinsics_cut_top_bottom()` | `--use_top_bottom_cut --cut_ratio X` | Sorts cameras by score, removes the top X% and bottom X%, then samples uniformly from the middle. Recommended for training (default `X=0.3`) |
| **Random** | `calculate_random_camera_extrinsics()` | `--use_random_pose` | Samples uniformly from all cameras, ignoring scores. Ablation baseline |

All three functions share the same signature:

```python
def strategy(scene_name, object_id, filemap, pose_data_path_base, cut_ratio) -> np.ndarray:
    """Returns a 4x4 camera-to-world extrinsic matrix."""
```

### 4.2 Adapting Pose Selection

**File path pattern**: `load_filemap()` constructs NPZ paths as:
```python
f"{filepath_base}/{scene_name}_view_instance_intersection_with_visibility.npz"
```
If your naming convention differs, modify this function.

**Pose file loading**: all three functions load the selected pose from disk:
```python
np.loadtxt(os.path.join(pose_data_path_base, scene_name, "pose", pose_filenames[idx]))
```
Adapt this path construction to your directory layout.

**Object ID = -1 fallback**: when `object_id == -1` (annotation has no associated object), the code falls back to a random valid pose. This handles datasets like SQA3D where the question is about the scene rather than a specific object.

---

## 5. Stage 3: PoseAlign — Point Cloud Transformation

Once a camera pose is selected, PoseAlign injects it into the model through one (or more) of three pathways.

### 5.1 PoseAlign-Transform (Recommended)

**What it does**: transforms the entire point cloud from world coordinates into the camera's egocentric frame, using a Forward-Left-Up (FLU) convention.

**Core class**: `ShiftCenterToCameraPose` in `llava/pc_utils/transform.py` (line 2537)

**Steps**:
1. Compute world-to-camera: `T_cw = inv(T_wc)`
2. Transform all points: `coord_cam = coord_world @ T_cw^T`
3. Apply RDF-to-FLU conversion (see [Section 8](#8-coordinate-system-reference)):

```
┌ 0  0  1  0 ┐
│-1  0  0  0 │    (Right→-Left, Down→-Up, Forward→Forward)
│ 0 -1  0  0 │
└ 0  0  0  1 ┘
```

**Effect on augmentation**: since the egocentric frame already encodes orientation, random rotation is disabled (`rot_range=[0, 0]`) and scene alignment is removed. Scaling and translation augmentations remain active.

**Adapting to your dataset**: if your camera coordinate system is not RDF (Right-Down-Forward), you need to modify the RDF-to-FLU conversion matrix. See [Section 8](#8-coordinate-system-reference) for common conventions.

### 5.2 PoseAlign-Prompt

**What it does**: converts the 4x4 camera extrinsic into a human-readable text description and inserts it into the conversation.

**Core method**: `encode_pose_prompt()` in `llava/train/train.py` (line 917)

The encoding extracts orientation vectors from the extrinsic matrix:
```python
position = pose[:3, 3]           # camera position in world frame
front    = pose[:3, 2]           # third column = forward (Z axis of camera)
left     = -pose[:3, 0]          # negated first column = left
up       = -pose[:3, 1]          # negated second column = up
```

These are formatted as: `"position [x,y,z], front: [fx,fy,fz], left: [lx,ly,lz], up: [ux,uy,uz]"` and injected into the conversation after the `<pc>` token via `add_pose_align_prompt()`.

**Adapting to your dataset**: this pathway operates directly on the 4x4 extrinsic. If your camera convention differs from RDF, adjust the column extraction above (the signs and indices of position/front/left/up). See [Section 8](#8-coordinate-system-reference).

### 5.3 PoseAlign-Embed

**What it does**: computes a 6D feature vector per point and concatenates it to the point cloud tokens before the multimodal projector.

**Core method**: `encode_camera_extrinsic_in_coords()` in `llava/model/llava_arch.py` (line 254)

The 6 dimensions are:

| Dim | Symbol | Description |
|---|---|---|
| 0 | `Xc` | X coordinate in camera frame |
| 1 | `Yc` | Y coordinate in camera frame |
| 2 | `Zc` | Z coordinate in camera frame (depth) |
| 3 | `u_norm` | Normalized horizontal image coordinate (`u / width`), or `-1` if out of frustum |
| 4 | `v_norm` | Normalized vertical image coordinate (`v / height`), or `-1` if out of frustum |
| 5 | `vis` | Binary visibility mask (1.0 = in frustum, 0.0 = outside) |

**Adapting to your dataset**: camera intrinsics are **hardcoded** at lines 257–258 of `llava_arch.py`. You must replace them:

```python
# Original (ScanNet):
fx, fy, cx, cy = 1169.621094, 1167.105103, 646.295044, 489.927032
width, height = 1296, 968

# Replace with your intrinsics:
fx, fy, cx, cy = YOUR_FX, YOUR_FY, YOUR_CX, YOUR_CY
width, height = YOUR_WIDTH, YOUR_HEIGHT
```

**Important**: this pathway widens the multimodal projector's input by 6 dimensions. A model trained with PoseAlign-Embed cannot load pretrained projector weights directly. You must merge weights with `merge_lora_model.py` before evaluation.

---

## 6. Adapting Training

### 6.1 Annotation Requirements

Each training sample must include a resolvable `object_id` that maps to an instance ID in the point cloud's `instance_gt` array. The training code resolves object IDs in `__getitem__()` (in `llava/train/train.py`, lines 1004–1026):

- **Single integer**: used directly.
- **List of integers**: one is chosen at random (for multi-object tasks like Multi3DRefer).
- **Empty or absent**: falls back to `object_id = -1` (random pose).

Since ScanQA in 3D-LLaVA's annotations do not include object IDs (which is more likely forgotten rather than intended), we create a supplementary JSON file mapping to supplement that information, which uses the `--extra_data_file` option.

### 6.2 Training Arguments

The full set of PoseAlign-specific training arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_cam_instance_intersect` | bool | `False` | Master switch: enable pose selection from PoseRecover data |
| `--apply_pose_to_pc` | bool | `False` | PoseAlign-Transform: transform point cloud to egocentric frame |
| `--apply_pose_to_prompt` | bool | `False` | PoseAlign-Prompt: inject pose text into conversation |
| `--apply_pose_to_projection` | bool | `False` | PoseAlign-Embed: concatenate 6D features to point tokens |
| `--use_top_bottom_cut` | bool | `False` | Use Clip strategy for pose selection |
| `--use_random_pose` | bool | `False` | Use Random strategy for pose selection (ablation) |
| `--cut_ratio` | float | `0.2` | Fraction to cut from top/bottom when using Clip strategy |
| `--pose_data_path_base` | str | `"playground/poses/scans"` | Directory containing camera pose files |
| `--extra_data_file` | str | `"playground/data/complementary_info/..."` | Supplementary object ID mapping (for datasets like ScanQA) |

### 6.3 Transform Pipeline Selection

Training automatically selects the correct transform pipeline based on argument flags. The logic is in `_build_pc_transform()` (`llava/train/train.py`, line 811):

| `apply_pose_to_pc` | `apply_pose_to_projection` | Pipeline | What differs from upstream |
|---|---|---|---|
| `False` | `False` | Upstream (original) | No pose involvement |
| `True` | `False` | `*_posealign` | Adds `ShiftCenterToCameraPose`; disables rotation |
| `False` | `True` | `*_collectPose` | Same as upstream but passes `possible_camera_extrinsic` through the pipeline |
| `True` | `True` | `*_posealign_collectpose` | Both: shifts PC and passes extrinsic |

Each category has 3 task variants: `referseg_*`, `vqa_*`, `densecap_*` (for referring segmentation, VQA, and dense captioning respectively), with both train and eval versions. These are defined in `llava/pc_utils/default.py`.

**Adding a new task type**: copy an existing pipeline (e.g., `vqa_transform_train_posealign`), change the `condition` string in the `Add` step, and adjust the `Collect` keys for any additional fields your task requires.

### 6.4 Pose Augmentation

The optional arguments `--pose_aug_rot_std` and `--pose_aug_trans_std` control random perturbation of selected poses during training. This is implemented by `perturb_pose_yaw_only()` in `llava/pc_utils/misc.py`:

- **Yaw rotation**: random rotation around the vertical axis within `[-std, +std]` radians
- **Translation**: Gaussian noise with the given standard deviation added to the camera position

This improves robustness to imperfect pose estimation but is not enabled by default (`std=0.0`).

---

## 7. Adapting Evaluation

### 7.1 Structure

Each benchmark has a PoseAlign evaluation script in `llava/eval/`:

| File | Benchmark | Task Type | Key Specialization |
|---|---|---|---|
| `model_scanqa_poseAlign.py` | ScanQA | VQA | Uses supplementary object ID mapping |
| `model_sqa3d_poseAlign.py` | SQA3D | VQA | Always `object_id=-1` (scene-level question) |
| `model_scanrefer_poseAlign.py` | ScanRefer | Referring Seg | Computes IoU at 0.25/0.50 |
| `model_scan2cap_poseAlign.py` | Scan2Cap | Dense Captioning | Loads Mask3D detection masks |
| `model_multi3drefer_poseAlign.py` | Multi3DRefer | Referring Seg | Handles multiple object IDs |

All share a common pattern:

1. Load model and tokenizer
2. For each sample: load scene data, call `get_possible_pose()`, apply transforms, generate output
3. Save predictions to JSON

### 7.2 Key Adaptation Points

**`get_possible_pose()`**: each eval script defines this function, mirroring the training-time pose selection logic. Adapt the object ID resolution for your annotation format.

**Transform selection**: same 4-category logic as training (see [Section 6.3](#63-transform-pipeline-selection)), but using eval-time transforms (no augmentation).

---

## 8. Coordinate System Reference

### 8.1 Conventions Used in This Codebase

| Frame | Convention | Axes |
|---|---|---|
| ScanNet world | Arbitrary (axis-aligned after scene alignment) | — |
| ScanNet camera | **RDF** | X = Right, Y = Down, Z = Forward |
| PoseAlign egocentric | **FLU** | X = Forward, Y = Left, Z = Up |

### 8.2 Extrinsic Matrix Convention

ScanNet pose files store **camera-to-world** transforms `T_wc`:

```
T_wc = [ R | t ]    where  point_world = T_wc @ point_camera
       [ 0 | 1 ]
```


### 8.3 RDF-to-FLU Conversion

The `ShiftCenterToCameraPose` transform applies this conversion after moving points to camera space:

```
        Camera (RDF)          Egocentric (FLU)
        X = Right      →     Y = -Right = Left
        Y = Down       →     Z = -Down  = Up
        Z = Forward    →     X = Forward
```

As a matrix:
```
M_rdf_to_flu = ┌ 0  0  1  0 ┐
               │-1  0  0  0 │
               │ 0 -1  0  0 │
               └ 0  0  0  1 ┘
```

### 8.4 Adapting for Other Camera Conventions

If your dataset uses a different camera coordinate system:

| Your Convention | Camera Axes | RDF-to-FLU Matrix Adjustment |
|---|---|---|
| **RDF** (ScanNet, most RGB-D) | X=Right, Y=Down, Z=Forward | No change needed |
| **RUB** (OpenGL, Blender) | X=Right, Y=Up, Z=Backward | Negate Z before converting; adjust matrix to `[[0,0,-1,0],[−1,0,0,0],[0,1,0,0],[0,0,0,1]]` |
| **FLU** (ROS) | X=Forward, Y=Left, Z=Up | No conversion needed — skip the RDF-to-FLU step entirely |
| **RDF** but transposed | `T_wc` stores world-to-camera instead of camera-to-world | Remove the `inv()` call in `ShiftCenterToCameraPose` |

To verify your convention is correct: after applying the full transform, objects in front of the camera should have positive X coordinates, objects to the left should have positive Y, and the floor should have negative Z.

### 8.5 PoseAlign-Prompt Column Extraction

The `encode_pose_prompt()` method extracts orientation vectors from the camera-to-world matrix assuming RDF:

```python
position = T_wc[:3, 3]     # translation = camera position in world
front    = T_wc[:3, 2]     # 3rd column = camera Z = forward
left     = -T_wc[:3, 0]    # negated 1st column = -right = left
up       = -T_wc[:3, 1]    # negated 2nd column = -down = up
```

For a different camera convention, adjust which columns map to front/left/up and whether negation is needed.

---

## 9. Porting Checklist

A step-by-step checklist for applying PoseAlign to a new dataset:

### Data Preparation

- [ ] Prepare point cloud scenes as `.pth` files with `coord` (N,3) and `instance_gt` (N,) arrays
- [ ] Extract camera poses as 4x4 camera-to-world matrices, one per video frame
- [ ] Record your camera intrinsics: `fx, fy, cx, cy, width, height`
- [ ] Confirm your camera coordinate convention (RDF, RUB, or other — see [Section 8.4](#84-adapting-for-other-camera-conventions))

### PoseRecover

- [ ] Adapt `pose_recover/export_pose_intersections_with_visibility_check.py`:
  - [ ] Replace camera intrinsics (line 71)
  - [ ] Adjust `pose_sample_interval` for your frame rate (line 74)
  - [ ] Adapt `load_scan_data()` to your point cloud format
  - [ ] Adapt pose file loading to your directory structure
- [ ] Run PoseRecover on all scenes
- [ ] Verify output: check non-zero visibility rates with `examine_scannet_intersect_metrics.py`

### Model Code

- [ ] If using PoseAlign-Embed: replace hardcoded intrinsics in `llava/model/llava_arch.py` (line 257)
- [ ] If your camera convention is not RDF: modify `ShiftCenterToCameraPose` conversion matrix and `encode_pose_prompt()` column extraction

### Training

- [ ] Ensure each annotation sample has a resolvable `object_id` (or create a supplementary mapping file)
- [ ] Adapt `load_filemap()` path pattern if your NPZ naming differs
- [ ] Adapt pose file path construction in `cam_instance_intersect.py` if your directory layout differs
- [ ] Create a training script with the appropriate PoseAlign flags (see [Section 6.2](#62-training-arguments))

### Evaluation

- [ ] Create evaluation model script following `model_*_poseAlign.py` pattern
- [ ] Create evaluation shell scripts with matching PoseAlign flags
- [ ] If using PoseAlign-Embed: run `merge_lora_model.py` before evaluation
