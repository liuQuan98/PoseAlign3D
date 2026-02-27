# !/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

EXECUTABLE=llava.eval.model_scan2cap_poseAlign
EXP_NAME=finetune-3d-llava-lora-PoseAlign-proj
MODEL_NAME=checkpoints/finetune-3d-llava-lora-PoseAlign-proj

MODEL_MERGED=${MODEL_NAME}-merged

# pay extra note: if we use proj layer poseAlign, we must use the merged model for evaluation
# REMEMBER TO DISABLE MODEL_BASE IN THIS CASE BECAUSE THE MODEL IS ALREADY COMPLETE AND DO NOT NEED A BASE MODEL

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m $EXECUTABLE \
        --scan-folder ./playground/data/scannet \
        --mask3d-inst-folder ./playground/data/eval_info/densecap_scanrefer/mask3d_inst_seg \
        --model-path $MODEL_MERGED \
        --question-file ./playground/data/eval_info/densecap_scanrefer/scan2cap_mask3d_val.json \
        --answers-file ./playground/predictions/${EXP_NAME}/densecap_scanrefer/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 \
        --use-top-bottom-cut False \
        --cut-ratio 0.2 \
        --pose-data-path-base "playground/poses/scans" \
        --extra-data-file "playground/data/complementary_info/matched_ScanQA_v1.0_val.json" \
        --apply-pose-to-pc False \
        --apply-pose-to-prompt False \
        --apply-pose-to-projection True &
done

wait

output_file=./playground/predictions/${EXP_NAME}/densecap_scanrefer/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/predictions/${EXP_NAME}/densecap_scanrefer/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python llava/eval/eval_scan2cap.py \
--pred-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_mask3d_val_attributes.pt \
--gt-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_val_attributes.pt \
--annotation-file ./playground/data/eval_info/densecap_scanrefer/scan2cap_val_corpus.json \
--result-file $output_file \



# Recently, we check the open source code of Video-3DLLM, finding the evaluation metric of Video-3D-LLM is slightly different from ours. Specifically, Video-3DLLM does not exclude "." and "," when compute the scores, resulting in higher result. We also provide the same evaluation to benefit the following works for conducting comparison under the same metric

# python llava/eval/eval_scan2cap_video3dllm_type.py \
# --pred-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_mask3d_val_attributes.pt \
# --gt-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_val_attributes.pt \
# --annotation-file ./playground/data/eval_info/densecap_scanrefer/scan2cap_val_corpus.json \
# --result-file $output_file
