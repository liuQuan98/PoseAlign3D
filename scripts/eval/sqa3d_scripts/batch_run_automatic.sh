# !/bin/bash

mkdir -p logs/eval_sqa3d

shopt -s nullglob  # no error if no matches
# for EVAL_SCRIPT_PATH in scripts/eval/sqa3d_scripts/multigpu_*.sh; do
for EVAL_SCRIPT_PATH in scripts/eval/sqa3d_scripts/multigpu_eval_sqa3d.sh; do
    echo "Processing: $EVAL_SCRIPT_PATH"
    filename=$(basename "$EVAL_SCRIPT_PATH" .sh)
    nohup bash $EVAL_SCRIPT_PATH > logs/eval_sqa3d/${filename}.txt &

    wait
done