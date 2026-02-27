#!/bin/bash

export THREAD_IDS=0,1,2,3
export PYTHONPATH=$(pwd)

thread_list="${THREAD_IDS:-0}"
IFS=',' read -ra THREADLIST <<< "$thread_list"

CHUNKS=${#THREADLIST[@]}

# if you want to process new data with old model, we usually output the inference results to new folders with the following postfix.
POSTFIX=''

EXECUTABLE=llm_as_judge/llm_judge_for_ScanQA_gpt5.py

specific_paths=("playground/predictions/finetune-3d-llava-lora/scanqa" "playground/predictions/finetune-3d-llava-lora-PoseAlign-pc-cut03-noFlip/scanqa")

shopt -s nullglob  # no error if no matches
for IDX_MODEL in $(seq 0 1); do
    PREDICTION_FILE_PATH=${specific_paths[$IDX_MODEL]}

    echo "Processing: $PREDICTION_FILE_PATH"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        echo "Processing chunk $IDX"
        python $EXECUTABLE \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --prediction-file-path $PREDICTION_FILE_PATH &
    done

    wait

    output_file=${PREDICTION_FILE_PATH}/llm_as_judge_gpt5/llm_as_judge_merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${PREDICTION_FILE_PATH}/llm_as_judge_gpt5/llm_as_judge_${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    wait
    python llm_as_judge/calculate_file_acc.py \
        --input_file $output_file
done
