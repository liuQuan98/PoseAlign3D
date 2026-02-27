import argparse
from llava.mm_utils import get_model_name_from_path
import torch, os
import time


import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.multimodal_encoder.pointcloud_encoder import SPConvPointCloudTower


def load_pretrained_model(model_path, model_base, model_name, pointcloud_tower_name="./checkpoints/pc_pretrained/ost-sa-only-llava-align-scannet200.pth", 
                          load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    if device_map is not None:
        kwargs = {"device_map": device_map, **kwargs}
    else:
        kwargs = {**kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)

            additional_encoder_cfg = {  
                "pc_feature_dim": 1024,
                "pc_hidden_size": 1024,
                "mm_hidden_size": 1030,
                "mm_inst_prompt_encoder": "shared_projector",
                "mm_patch_merge_type": "flat",
                "mm_pointcloud_tower": "./checkpoints/pc_pretrained/ost-sa-only-llava-align-scannet200.pth",
                "mm_projector_lr": None,
            }
            assert "mm_inst_prompt_encoder" not in lora_cfg_pretrained.to_dict()
            assert "mm_patch_merge_type" not in lora_cfg_pretrained.to_dict()
            assert "mm_pointcloud_tower" not in lora_cfg_pretrained.to_dict()

            lora_cfg_pretrained.update(additional_encoder_cfg)

            import copy
            lora_cfg_original_vocab_size = copy.deepcopy(lora_cfg_pretrained)
            if 'checkpoints' in model_path.lower():
                print("using vocab size 32000 to initialize model", flush=True)
                lora_cfg_original_vocab_size.vocab_size = lora_cfg_pretrained.vocab_size - 1 # 32001-1 = 32000
                assert lora_cfg_original_vocab_size.vocab_size == 32000

            if os.path.exists(os.path.join(model_path, 'tokenizer.model')):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage='device_map' in kwargs, config=lora_cfg_original_vocab_size, **kwargs)
            token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features

            if 'checkpoints' in model_path.lower():
                token_num = lora_cfg_pretrained.vocab_size
                print(f"Resizing lm_head and embed_tokens from {model.lm_head.weight.shape[0]} to {token_num}")
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
                model.config.vocab_size = token_num

            pointcloud_tower = model.get_pointcloud_tower()
            
            if pointcloud_tower is None or not pointcloud_tower.is_loaded:
                pointcloud_tower.load_model(pointcloud_tower_name=pointcloud_tower_name)
                pointcloud_tower.to(device=model.device, dtype=torch.bfloat16)

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}

            for k, v in non_lora_trainables.items():
                if 'lm_head' in k:
                    print(f"found lm_head in non_lora_trainables and overridden correctly", flush=True)
                elif 'embed_tokens' in k:
                    print(f"found model.embed_tokens in non_lora_trainables and overridden correctly", flush=True)
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            missing_keys, unexpected_keys = model.load_state_dict(non_lora_trainables, strict=False)
            print("Existing keys in non_lora_trainables:", list(non_lora_trainables.keys()), flush=True)
            print("Missing keys when loading non_lora_trainables:", missing_keys, " Unexpected keys:", unexpected_keys, flush=True)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, None, context_len


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    t1 = time.time()
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map=None, pointcloud_tower_name=None, ignore_mismatched_sizes=True)
    
    t2 = time.time()
    print(f"Time to load model: {t2 - t1} seconds")
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    torch.save(model.get_model().mm_projector.state_dict(), os.path.join(args.save_model_path, 'mm_projector.bin'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/finetune-3d-llava-lora-PoseAlign-proj")
    parser.add_argument("--model-base", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--save-model-path", type=str, default="checkpoints/finetune-3d-llava-lora-PoseAlign-proj-merged")

    args = parser.parse_args()

    merge_lora(args)