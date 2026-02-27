# !/bin/bash

mkdir -p logs/eval_multi3dref

shopt -s nullglob  # no error if no matches
# for EVAL_SCRIPT_PATH in scripts/eval/multi3dref_scripts/multigpu_*.sh; do
for EVAL_SCRIPT_PATH in scripts/eval/multi3dref_scripts/multigpu_eval_multi3drefer_poseAlign_proj.sh; do
    echo "Processing: $EVAL_SCRIPT_PATH"
    filename=$(basename "$EVAL_SCRIPT_PATH" .sh)
    nohup bash $EVAL_SCRIPT_PATH > logs/eval_multi3dref/${filename}.txt &

    wait
done