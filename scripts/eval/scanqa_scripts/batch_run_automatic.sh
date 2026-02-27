# !/bin/bash

mkdir -p logs/eval_scanqa

shopt -s nullglob  # no error if no matches
for EVAL_SCRIPT_PATH in scripts/eval/scanqa_scripts/multigpu_*.sh; do
    echo "Processing: $EVAL_SCRIPT_PATH"
    filename=$(basename "$EVAL_SCRIPT_PATH" .sh)
    nohup bash $EVAL_SCRIPT_PATH > logs/eval_scanqa${filename}.txt &

    wait
done