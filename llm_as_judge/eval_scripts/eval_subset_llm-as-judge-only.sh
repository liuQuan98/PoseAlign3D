#!/bin/bash

dataset=scanqa

files=('' '-noPoseAlign-trainSampler' '-PoseAlign-pc-cut00-noFlip' '-PoseAlign-pc-cut01-noFlip' '-PoseAlign-pc-cut02-noFlip' '-PoseAlign-pc-cut03-noFlip' '-PoseAlign-pc-cut04-noFlip' '-PoseAlign-pc-cut045-noFlip' '-PoseAlign-pc-cut049-noFlip' '-PoseAlign-pc-trainSampler' '-PoseAlign-pc' '-PoseAlign-proj' '-PoseAlign-prompt')

for suffix in "${files[@]}"; do
    LLM_AS_JUDGE_FILE_PATH=playground/predictions/finetune-3d-llava-lora${suffix}/${dataset}/llm_as_judge/llm_as_judge_merge.jsonl

    python llm_as_judge/calculate_direction_critical_subset_metrics.py \
        --annotation-file foo \
        --use-complementary-subset False \
        --llm-only True \
        --llm-as-judge-file $LLM_AS_JUDGE_FILE_PATH

    wait
done
