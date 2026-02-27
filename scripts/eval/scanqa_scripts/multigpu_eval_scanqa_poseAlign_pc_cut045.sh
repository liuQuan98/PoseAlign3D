# !/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

EXECUTABLE=llava.eval.model_scanqa_poseAlign
EXP_NAME=finetune-3d-llava-lora-PoseAlign-pc-cut045
MODEL_NAME=checkpoints/finetune-3d-llava-lora-PoseAlign-pc-cut045

MODEL_MERGED=${MODEL_NAME}-merged

# pay extra note: if we use proj layer poseAlign, we must use the merged model for evaluation
# for example, run python merge_lora_model.py

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m $EXECUTABLE \
        --scan-folder ./playground/data/scannet/val \
        --model-path $MODEL_NAME \
        --model-base liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval_info/scanqa/scanqa_val_question.jsonl \
        --answers-file ./playground/predictions/$EXP_NAME/scanqa/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --use-top-bottom-cut True \
        --cut-ratio 0.49 \
        --pose-data-path-base "playground/poses/scans" \
        --extra-data-file "playground/data/complementary_info/matched_ScanQA_v1.0_val.json" \
        --apply-pose-to-pc True \
        --apply-pose-to-prompt False \
        --apply-pose-to-projection False &
done

wait

output_file=./playground/predictions/$EXP_NAME/scanqa/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/predictions/$EXP_NAME/scanqa/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_scanqa.py \
    --annotation-file ./playground/data/eval_info/scanqa/scanqa_val_answer.jsonl \
    --result-file $output_file
