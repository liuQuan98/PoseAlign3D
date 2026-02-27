#!/bin/bash

python ./llm_as_judge/evaluate_opposite_yaw.py --training_file_path playground/data/train_info/multi3drefer_train_3d_llava.json

wait

python ./llm_as_judge/evaluate_opposite_yaw.py --training_file_path playground/data/train_info/nr3d_caption_train_3d_llava.json

wait

python ./llm_as_judge/evaluate_opposite_yaw.py --training_file_path playground/data/train_info/sr3d_train_3d_llava.json

