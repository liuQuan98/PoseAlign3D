#! /bin/bash

bash llm_as_judge/llm_as_judge_scripts/batch_llm_as_judge_scan2cap.sh

wait

bash llm_as_judge/llm_as_judge_scripts/batch_llm_as_judge_scanqa.sh

wait

bash llm_as_judge/llm_as_judge_scripts/batch_llm_as_judge_sqa3d.sh


# bash llm_as_judge/llm_as_judge_scripts/batch_llm_as_judge_scan2cap_gpt5.sh

# wait

# bash llm_as_judge/llm_as_judge_scripts/batch_llm_as_judge_scanqa_gpt5.sh

# wait

# bash llm_as_judge/llm_as_judge_scripts/batch_llm_as_judge_sqa3d_gpt5.sh