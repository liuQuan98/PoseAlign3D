#!/bin/bash

EXP_NAME=finetune-3d-llava-lora-PoseAlign-pc-cut03-noFlip

output_file=./playground/predictions/$EXP_NAME/scanqa/merge.jsonl
llm_as_judge_file=./playground/predictions/$EXP_NAME/scanqa/llm_as_judge/llm_as_judge_merge.jsonl

python llm_as_judge/calculate_direction_critical_subset_metrics.py \
    --annotation-file ./playground/data/eval_info/scanqa/scanqa_val_answer.jsonl \
    --question-file ./playground/data/eval_info/scanqa/scanqa_val_question.jsonl \
    --subset-file ./llm_as_judge/scanqa_vllm/merge.jsonl \
    --result-file $output_file \
    --use-llm-decided-subset True \
    --use-complementary-subset True \
    --llm-as-judge-file $llm_as_judge_file
