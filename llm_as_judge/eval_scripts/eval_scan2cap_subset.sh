#!/bin/bash

EXP_NAME=finetune-3d-llava-lora

output_file=./playground/predictions/$EXP_NAME/densecap_scanrefer/merge.jsonl
llm_as_judge_file=./playground/predictions/$EXP_NAME/densecap_scanrefer/llm_as_judge/llm_as_judge_merge.jsonl

python llm_as_judge/calculate_direction_critical_subset_metrics.py \
    --pred-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_mask3d_val_attributes.pt \
    --gt-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_val_attributes.pt \
    --annotation-file ./playground/data/eval_info/densecap_scanrefer/scan2cap_val_corpus.json \
    --subset-file ./llm_as_judge/scan2cap_vllm_corpus/merge.jsonl \
    --result-file $output_file \
    --use-llm-decided-subset True \
    --use-complementary-subset False \
    --llm-as-judge-file $llm_as_judge_file
