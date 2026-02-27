import json
import numpy as np


if __name__ == "__main__":
    with open("playground/data/complementary_info/ScanQA_v1.0_train.json", 'r') as f:
        ll3da_data = json.load(f)
    
    with open("playground/data/train_info/scanqa_train_3d_llava.json", 'r') as f:
        threedllava_data = json.load(f)

    ll3da_data_questions = [item['question'] for item in ll3da_data]
    ll3da_object_ids = [item['object_ids'] for item in ll3da_data]
    ll3da_scene_ids = [item['scene_id'] for item in ll3da_data]
    threedllava_questions = [item['conversations'][0]['value'][6:][:-28] for item in threedllava_data] # remove "<pc>\n " prefix and " Answer the question simply." suffix
    threedllava_scene_ids = [item['scene_id'] for item in threedllava_data]

    retrieved_object_ids = []
    matched_count = 0
    for question, scene_id in zip(threedllava_questions, threedllava_scene_ids):

        scene_matched_ids = [i for i, item in enumerate(ll3da_scene_ids) if item == scene_id]

        for idx in scene_matched_ids:
            if ll3da_data_questions[idx] == question:
                retrieved_object_ids.append(ll3da_object_ids[idx])
                matched_count += 1
                break
        # if question in ll3da_data_questions:
        #     retrieved_object_ids.append(ll3da_object_ids[ll3da_data_questions.index(question)])
        #     matched_count += 1
        else:
            retrieved_object_ids.append([-1]) # -1 means not found
    
    print(f"Total LL3DA ScanQA data: {len(ll3da_data_questions)}")
    print(f"Total 3DLLAVA ScanQA data: {len(threedllava_questions)}")
    print(f"Total matched data: {matched_count}")

    with open("playground/data/complementary_info/matched_ScanQA_v1.0_train.json", 'w') as f:
        json.dump(retrieved_object_ids, f, indent=4)


    # validation set
    with open("playground/data/complementary_info/ScanQA_v1.0_val.json", 'r') as f:
        ll3da_data = json.load(f)

    threedllava_data = [json.loads(q) for q in open("playground/data/eval_info/scanqa/scanqa_val_question.jsonl", "r")]

    ll3da_scene_ids = [item['scene_id'] for item in ll3da_data]
    ll3da_question_ids = [item['question_id'] for item in ll3da_data]
    ll3da_object_ids = [item['object_ids'] for item in ll3da_data]
    threedllava_question_ids = [item['question_id'] for item in threedllava_data]

    retrieved_object_ids = {}
    matched_count = 0
    for question_id in threedllava_question_ids:

        question_matched_ids = [i for i, item in enumerate(ll3da_question_ids) if item == question_id]

        assert len(question_matched_ids) == 1, f"question id {question_id} has multiple matches in ll3da data, please check."

        idx = question_matched_ids[0]
        retrieved_object_ids[question_id] = ll3da_object_ids[idx]
        matched_count += 1
        # if question in ll3da_data_questions:
        #     retrieved_object_ids.append(ll3da_object_ids[ll3da_data_questions.index(question)])
        #     matched_count += 1
    
    print(f"Total LL3DA ScanQA data: {len(ll3da_question_ids)}")
    print(f"Total 3DLLAVA ScanQA data: {len(threedllava_question_ids)}")
    print(f"Total matched data: {matched_count}")

    with open("playground/data/complementary_info/matched_ScanQA_v1.0_val.json", 'w') as f:
        json.dump(retrieved_object_ids, f, indent=4)
    