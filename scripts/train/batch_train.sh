#!/bin/bash

bash scripts/train/finetune-3d-llava-lora_poseAlign-pc-cut00.sh
wait
bash scripts/train/finetune-3d-llava-lora_poseAlign-pc-cut02.sh
wait
bash scripts/train/finetune-3d-llava-lora_poseAlign-pc-cut03.sh
wait
bash scripts/train/finetune-3d-llava-lora_poseAlign-pc-cut04.sh
wait
bash scripts/train/finetune-3d-llava-lora_poseAlign-pc-cut045.sh
wait
bash scripts/train/finetune-3d-llava-lora_poseAlign-pc-cut049.sh