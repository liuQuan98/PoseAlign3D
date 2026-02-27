# !/bin/bash

mkdir -p logs/eval_scan2cap

shopt -s nullglob  # no error if no matches
# for EVAL_SCRIPT_PATH in scripts/eval/scan2cap_scripts/multigpu_*.sh; do
for EVAL_SCRIPT_PATH in scripts/eval/scan2cap_scripts/multigpu_eval_scan2cap_poseAlign_proj.sh; do
    echo "Processing: $EVAL_SCRIPT_PATH"
    filename=$(basename "$EVAL_SCRIPT_PATH" .sh)
    nohup bash $EVAL_SCRIPT_PATH > logs/eval_scan2cap/${filename}.txt &

    wait
done