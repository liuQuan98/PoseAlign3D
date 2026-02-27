#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

EXECUTABLE=llm_as_judge/llm_judge_for_LR_vllm.py

FOLDER_NAME=./llm_as_judge/scanqa_vllm


for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Processing chunk $IDX on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python $EXECUTABLE \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=${FOLDER_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${FOLDER_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
