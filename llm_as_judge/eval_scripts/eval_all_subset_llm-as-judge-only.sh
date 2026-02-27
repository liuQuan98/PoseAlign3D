#!/bin/bash

for LLM_AS_JUDGE_FILE_PATH in playground/predictions/*/{scanqa,sqa3d,densecap_scanrefer}/llm_as_judge/llm_as_judge_merge.jsonl; do

    echo "Processing: $LLM_AS_JUDGE_FILE_PATH"
    python llm_as_judge/calculate_direction_critical_subset_metrics.py \
        --use-llm-decided-subset True \
        --use-complementary-subset False \
        --subset-file ./llm_as_judge/scanqa_vllm/merge.jsonl \
        --llm-only True \
        --llm-as-judge-file $LLM_AS_JUDGE_FILE_PATH

    wait
done
