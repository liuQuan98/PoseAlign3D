# !/bin/bash

mkdir -p logs/eval_scanrefer

shopt -s nullglob  # no error if no matches
for EVAL_SCRIPT_PATH in scripts/eval/scanrefer_scripts/multigpu_*.sh; do
    echo "Processing: $EVAL_SCRIPT_PATH"
    filename=$(basename "$EVAL_SCRIPT_PATH" .sh)
    nohup bash $EVAL_SCRIPT_PATH > logs/eval_scanrefer/${filename}.txt &

    wait
done