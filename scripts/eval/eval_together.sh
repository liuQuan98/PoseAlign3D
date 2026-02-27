#!/bin/bash

bash scripts/eval/multi3dref_scripts/batch_run_automatic.sh

wait 

bash scripts/eval/scan2cap_scripts/batch_run_automatic.sh

wait

bash scripts/eval/scanqa_scripts/batch_run_automatic.sh

wait

bash scripts/eval/scanrefer_scripts/batch_run_automatic.sh

wait

bash scripts/eval/sqa3d_scripts/batch_run_automatic.sh

wait


