import argparse
import torch
import os
import copy
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_LINK_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
# from llava.pc_utils import densecap_transform_eval, Compose
from PIL import Image
from llava.mm_utils import tokenizer_special_token
import math
import pathlib
import numpy as np
from llava.train.train import DataCollatorForSupervisedDataset
from pointgroup_ops import voxelization_idx
from typing import Dict, Optional, Sequence, List

from collections import defaultdict, OrderedDict

from llava.train.cam_instance_intersect import load_filemap, calculate_best_fit_camera_extrinsics, calculate_random_camera_extrinsics_cut_top_bottom, calculate_random_camera_extrinsics
from llava.pc_utils import densecap_transform_eval, densecap_transform_eval_posealign, densecap_transform_eval_collectPose, densecap_transform_eval_posealign_collectpose, Compose
from llava.constants import DEFAULT_PC_TOKEN, DEFAULT_LINK_TOKEN, LINK_TOKEN_INDEX, DEFAULT_POSE_TOKEN, DEFAULT_PC_PROMPT
from llava.pc_utils.misc import perturb_pose_yaw_only


template = [
    "<pc>\nDescribe this object <loc> in the given 3D scene."
]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def ponder_collate_fn(batch, max_point=-1):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    # we drop a large data if it exceeds max_point
    # note that directly drop the last one may cause problem
    if max_point > 0:
        accum_num_points = 0
        ret_batches = []
        for batch_id, data in enumerate(batch):
            num_coords = data["coord"].shape[0]
            if accum_num_points + num_coords > max_point:
                continue
            accum_num_points += num_coords
            ret_batches.append(data)
        return ponder_collate_fn(ret_batches)

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [ponder_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: ponder_collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)


def get_possible_pose(scene_name, args, object_id):
    # global extra_data_dict
    cam_instance_intersection_filemap = load_filemap(scene_name=scene_name)

    # this is not scanqa, follow standard protocol to choose one object id
    if isinstance(object_id, Sequence):
        object_id = np.array(object_id).reshape(-1)
        try:
            if object_id.size == 0:
                object_id_tmp = -1
            else:
                object_id_tmp = np.random.choice(object_id.reshape(-1))
        except Exception as e:
            print(f"Error in choosing object_id from {object_id}, scene {scene_name}: {e}", flush=True)
            raise e
    elif isinstance(object_id, int):
        object_id_tmp = object_id
    else:
        raise NotImplementedError("object_id should be int or list of int, but got {}".format(type(object_id)))

    if args.use_random_pose:
        pose_instance_process_function = calculate_random_camera_extrinsics
    elif args.use_top_bottom_cut:
        pose_instance_process_function = calculate_random_camera_extrinsics_cut_top_bottom
    else:
        pose_instance_process_function = calculate_best_fit_camera_extrinsics
    
    try:
        possible_camera_extrinsic = pose_instance_process_function(
            scene_name=scene_name, object_id=object_id_tmp, filemap=cam_instance_intersection_filemap,
            pose_data_path_base=args.pose_data_path_base, cut_ratio=args.cut_ratio
        )
        if np.isnan(possible_camera_extrinsic).any() or np.isinf(possible_camera_extrinsic).any():
            print(f"possible_camera_extrinsic contains NaN or INF: {possible_camera_extrinsic}", flush=True)
            raise ValueError
    except Exception as e:
        print(f"Error in calculating camera extrinsics for scene {scene_name}, object {object_id_tmp}: {e}", flush=True)
        raise e
    return possible_camera_extrinsic


def eval_model(args):
    # Model
    disable_torch_init()
    # model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, model, _, _ = load_pretrained_model(model_path, args.model_base, model_name, pointcloud_tower_name=args.pointcloud_tower_name, ignore_mismatched_sizes=True)
    
    if args.model_base is None:
        # the model is a merged model, no need to load lora weights
        model_name = get_model_name_from_path(args.model_path)
        model_name = model_name.replace('lora', '')
        # if args.apply_pose_to_projection:
        #     device_map = None 
        # else:
        #     device_map = "auto"
        device_map = "auto"
        tokenizer, model, _, _ = load_pretrained_model(args.model_path, None, model_name, pointcloud_tower_name=args.pointcloud_tower_name, device_map=device_map, ignore_mismatched_sizes=True)
    else:
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        if args.apply_pose_to_projection:
            # Note: when applying pose to projection, we need to turn off low_cpu_mem_usage and disable device_map.
            # Long story short, it causes bugs otherwise.
            # Specifically, the loading process first tries to load the whole base model and immediately dispatches it to the device_map before we can even load non_lora_trainables.bin
            # But the new projection layer is not in the base model, it is in non_lora_trainables.bin as we already modified it to a larger size
            # By default, when a new weight is not initialized and we allow low_cpu_mem_usage, transformers will not try to initialize the mismatched weights randomly.
            # This causes the model to contain some uninitialized weights, which will cause errors in dispatch process.
            device_map = None 
        else:
            device_map = "auto"
        tokenizer, model, _, _ = load_pretrained_model(model_path, args.model_base, model_name, pointcloud_tower_name=args.pointcloud_tower_name, device_map=device_map, ignore_mismatched_sizes=True)


    # eval info
    with open(args.question_file, "rb") as f:
        questions = json.load(f)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    record = dict()
    for idx, source in tqdm(enumerate(questions), total=len(questions)):
        scan_file = source["scene_id"]
        pred_id = source["pred_id"]
        
        scan_folder = args.scan_folder
        scan_data_path  = pathlib.Path(scan_folder) / 'val' / f'{scan_file}.pth'
        superpoint_path = pathlib.Path(scan_folder) / 'super_points' / f'{scan_file}.bin'

        raw_data = torch.load(scan_data_path)
        coord = raw_data['coord']
        color = raw_data['color']
        superpoint_mask = np.fromfile(superpoint_path, dtype=np.int64)

        pred_mask_path = pathlib.Path(args.mask3d_inst_folder) / f'{scan_file}.pt'
        fg_mask_ind = torch.load(pred_mask_path)[pred_id]['segments']
        fg_mask_ind = np.array(fg_mask_ind)
        instance = np.zeros(coord.shape[0])
        instance[fg_mask_ind] = 1
        instance = instance.astype(bool)

        # data transformation
        # transform = Compose(densecap_transform_eval)

        object_ids = int(source["obj_id"])
        possible_pose = get_possible_pose(scan_file, args, object_id=object_ids)
        if args.pose_aug_rot_std > 1e-6 or args.pose_aug_trans_std > 1e-6:
            possible_pose = perturb_pose_yaw_only(possible_pose, args.pose_aug_rot_std, args.pose_aug_trans_std, left_multiply=False)

        # data transformation
        if args.apply_pose_to_pc and not args.apply_pose_to_projection:
            transform = Compose(densecap_transform_eval_posealign)
        elif args.apply_pose_to_projection and not args.apply_pose_to_pc:
            transform = Compose(densecap_transform_eval_collectPose)
        elif args.apply_pose_to_pc and args.apply_pose_to_projection:
            transform = Compose(densecap_transform_eval_posealign_collectpose)
        else:
            transform = Compose(densecap_transform_eval)

        pc_data_dict = dict(
            scene_id=scan_file,
            coord=coord,
            color=color,
            instance=instance,
            superpoint_mask=superpoint_mask,
            possible_camera_extrinsic=possible_pose,
        )
        pc_data_dict = transform(pc_data_dict)

        grid_coord = pc_data_dict['grid_coord']
        grid_coord = torch.cat([torch.LongTensor(grid_coord.shape[0], 1).fill_(0), grid_coord], 1)
        pc_data_dict['grid_coord'] = grid_coord

        # voxelize
        batch_size = 1
        grid_coords = pc_data_dict["grid_coord"]
        spatial_shape = np.clip((grid_coords.max(0)[0][1:] + 1).numpy(), 128, None)  # long [3]
        voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coords, batch_size, 4)

        for key in pc_data_dict:
            if key in ["coord", "grid_coord", "feat", "offset", "obj_click", "obj_sp_mask", "possible_camera_extrinsic"]:
                pc_data_dict[key] = ponder_collate_fn([pc_data_dict[key]])

        qs = template[0].replace("<pc>", DEFAULT_IMAGE_TOKEN)

        # add the posealign prompt
        # DEFAULT_PC_PROMPT = f" You are an embodied agent posed at {DEFAULT_POSE_TOKEN}. The referred object is in front of you."
        if args.apply_pose_to_prompt:
            possible_pose = possible_pose
            # the pose is a 4x4 matrix derived from camera extrinsics directly, which has rdf coordinate system
            # we correct that into flu coordinate system for better human understanding
            position = possible_pose[:3, 3]
            front_direction = possible_pose[:3, 2]
            left_direction = -possible_pose[:3, 0]
            up_direction = -possible_pose[:3, 1]
            pose_encoded = f"position {position}, front: {front_direction}, left: {left_direction}, up: {up_direction}"
            encoded_pose_prompt = DEFAULT_PC_PROMPT.replace(DEFAULT_POSE_TOKEN, pose_encoded).strip()
            qs = encoded_pose_prompt + " " + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        device = model.device
        input_ids = tokenizer_special_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
        coord = pc_data_dict["coord"].to(device, dtype=torch.bfloat16)
        voxel_coords = voxel_coords.to(device)
        offset = pc_data_dict["offset"].to(device)
        feat = pc_data_dict["feat"].to(device, dtype=torch.bfloat16)
        p2v_map = p2v_map.to(device)
        v2p_map = v2p_map.to(device)
        superpoint_mask = [torch.tensor(superpoint_mask).to(device)]
        obj_click = pc_data_dict["obj_click"].to(device, dtype=torch.bfloat16)
        obj_click = obj_click.unsqueeze(0)
        obj_sp_mask = [pc_data_dict["obj_sp_mask"].to(device)]
        possible_pose = pc_data_dict["possible_camera_extrinsic"].unsqueeze(0).to(device, dtype=torch.bfloat16) if args.apply_pose_to_projection else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                click=obj_click,
                click_mask=obj_sp_mask,
                # images=image_tensor.unsqueeze(0).half().cuda(),
                # image_sizes=[image.size],
                coord=coord,
                grid_coord=voxel_coords,
                offset=offset,
                feat=feat,
                p2v_map=p2v_map,
                v2p_map=v2p_map,
                spatial_shape=spatial_shape,
                superpoint_mask=superpoint_mask,
                conditions=[pc_data_dict["condition"]],
                # do_sample=True if args.temperature > 0 else False,
                do_sample=False,
                num_beams=5,
                min_length=1,
                no_repeat_ngram_size=3,
                temperature=1.0,
                max_new_tokens=64,
                tokenizer=tokenizer,
                use_cache=True,
                possible_pose=possible_pose,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        
        ans_file.write(json.dumps({"scene_id": scan_file,
                                   "pred_id": pred_id,
                                   "gt_id": source["obj_id"],
                                   "prompt": prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--pointcloud-tower-name", type=str, default=None)
    parser.add_argument("--scan-folder", type=str, default="")
    parser.add_argument("--mask3d-inst-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--pointcloud_tower_name", type=str, default="./checkpoints/pc_pretrained/ost-sa-only-llava-align-scannet200.pth")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    # newly added args for poseAlign
    parser.add_argument("--extra-data-file", type=str, default="playground/data/complementary_info/matched_ScanQA_v1.0_val.json")
    parser.add_argument("--pose-data-path-base", type=str, default="playground/poses/scans")
    parser.add_argument("--use-top-bottom-cut", type=str2bool, default=False, help="whether to use top-bottom cut when calculating possible poses")
    parser.add_argument("--use-random-pose", type=str2bool, default=False, help="whether to use random poses when calculating possible poses")
    parser.add_argument("--cut-ratio", type=float, default=0.2, help="the cut ratio for top-bottom cut")
    parser.add_argument("--apply-pose-to-pc", type=str2bool, help="whether to apply pose information to the input point cloud. If False, use the original point cloud.")
    parser.add_argument("--apply-pose-to-prompt", type=str2bool, help="whether to apply pose information to the prompt. If False, use the original prompt.")
    parser.add_argument("--apply-pose-to-projection", type=str2bool, help="whether to inject pose information into the Encoder --> LLM projection layer. If False, use the original projection layer.")
    parser.add_argument("--pose-encode-dim", type=int, default=6)    
    parser.add_argument("--pose-aug-rot-std", type=float, default=0.)
    parser.add_argument("--pose-aug-trans-std", type=float, default=0.)

    args = parser.parse_args()

    eval_model(args)
