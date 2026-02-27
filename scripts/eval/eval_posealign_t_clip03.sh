#!/bin/bash

bash scripts/eval/multi3dref_scripts/multigpu_eval_multi3drefer_poseAlign_pc.sh

wait 

bash scripts/eval/scan2cap_scripts/multigpu_eval_scan2cap_poseAlign_pc.sh

wait

bash scripts/eval/scanqa_scripts/multigpu_eval_scanqa_poseAlign_pc.sh

wait

bash scripts/eval/scanrefer_scripts/multigpu_eval_scanrefer_poseAlign_pc.sh

wait

bash scripts/eval/sqa3d_scripts/multigpu_eval_sqa3d_poseAlign_pc.sh

wait


