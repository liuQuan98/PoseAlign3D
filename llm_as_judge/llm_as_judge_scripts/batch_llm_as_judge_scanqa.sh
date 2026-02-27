#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# if you want to process new data with old model, we usually output the inference results to new folders with the following postfix.
POSTFIX=''

EXECUTABLE=llm_as_judge/llm_judge_for_ScanQA_vllm.py

shopt -s nullglob  # no error if no matches
for PREDICTION_FILE_PATH in playground/predictions/*-randomPose/scanqa; do
    if [[ $PREDICTION_FILE_PATH == *"cut01"* ]]; then
        continue
    fi
    echo "Processing: $PREDICTION_FILE_PATH"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        echo "Processing chunk $IDX on GPU ${GPULIST[$IDX]}"
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python $EXECUTABLE \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --prediction-file-path $PREDICTION_FILE_PATH &
    done

    wait

    output_file=${PREDICTION_FILE_PATH}/llm_as_judge/llm_as_judge_merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${PREDICTION_FILE_PATH}/llm_as_judge/llm_as_judge_${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    wait
    python llm_as_judge/calculate_file_acc.py \
        --input_file $output_file
done
