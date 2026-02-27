import torch
import json
from tqdm import tqdm
import math
import argparse
import os
import json
from openai import OpenAI


with open('llm_as_judge/openai_key.txt', 'r') as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()


GRADER_TEMPLATE = """
Your job is to look at a set of ground-truth answers describing an object in the scene, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Ground truth: "There is a rectangular cabinet overhanging. It is next to a copier.",
"A broad black closed cabinet. It is attached to the wall.",
"A black broad cabinet. It is attached to the wall.",
"The cabinet is affixed to the wall. It is above the printer and the lower counter.",
"The cabinet is located on the left side of the room. It is affixed to the wall. It is above the printer below it."
Predicted answer 1: this is a black cabinet. it is to the right of the window.
Predicted answer 2: there is a rectangular cabinet. it is over a printer.
```
These predicted answers are all CORRECT because:
    - They fully contain the key information describing the referred target, which in this case is the cabinet. Aliases, plurals, and synonyms are acceptable.
    - They do not contain any information that contradicts the ground truth.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the ground truth is fully included and the response contains no incorrect information or contradictions.

    
The following are examples of INCORRECT predicted answers.
```
Ground truth: "There is a rectangular cabinet overhanging. It is next to a copier.",
"A broad black closed cabinet. It is attached to the wall.",
"A black broad cabinet. It is attached to the wall.",
"The cabinet is affixed to the wall. It is above the printer and the lower counter.",
"The cabinet is located on the left side of the room. It is affixed to the wall. It is above the printer below it."
Predicted answer 1: this is a white cabinet. it is to the right of the window.
Predicted answer 2: there is a rectangular cabinet. it is over a trash can.
Predicted answer 3: there is a bookshelf. it is attached to a wall.
```
These predicted answers are all INCORRECT because:
    - Some of the information in the predicted answer contradicts the ground truth. For example, "white" vs. "black", "trash can" vs. "printer".
    - Or the predicted answer conains the wrong object entirely, e.g., "bookshelf" instead of "cabinet".
    - Or the answer shies away from the question and does not provide any useful information.


Also note the following things:
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What surface texture is the wooden cardboard?" and the gold target "wooden matte", "brown flat". The predicted answer "flat" would be considered CORRECT, even though it does not include "wooden", since it is clear from the question that the cardboard is wooden.

    
Here is a new example. Simply reply with either CORRECT or INCORRECT.
```
Ground truth: {ground_truth}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
""".strip()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--prediction-file-path", type=str, required=True, default="playground/predictions/model_foo/scanqa")
    args = parser.parse_args()

    basepath = args.prediction_file_path
    assert os.path.exists(basepath), f"Path {basepath} does not exist."

    os.makedirs(os.path.join(basepath, "llm_as_judge_gpt5"), exist_ok=True)
    output_file = os.path.join(basepath, "llm_as_judge_gpt5", f"llm_as_judge_{args.num_chunks}_{args.chunk_idx}.jsonl")
    output_failed_ids = os.path.join(basepath, "llm_as_judge_gpt5", f"llm_as_judge_{args.num_chunks}_{args.chunk_idx}_failed_ids.txt")

    # --- 1) Reuse the model ---
    client = OpenAI()

    question_file = "playground/data/eval_info/densecap_scanrefer/scan2cap_mask3d_val.json"
    prediction_file = os.path.join(args.prediction_file_path, "merge.jsonl")

    with open(question_file, "rb") as f:
        questions = json.load(f)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    questions = {f"{item['scene_id']}_obj-id_{item['obj_id']}_pred-id_{item['pred_id']}": item['ref_captions'] for item in questions}
    predicted_answers = [json.loads(p) for p in open(os.path.expanduser(prediction_file), "r")]
    predicted_answers_map = {f"{p['scene_id']}_obj-id_{p['gt_id']}_pred-id_{p['pred_id']}": p['text'] for p in predicted_answers}

    qap_pairs = {k: [questions[k], predicted_answers_map[k]] for k in questions}

    judgements = {}
    failed_ids = []

    ans_file = open(output_file, "w")

    for qid in tqdm(qap_pairs.keys()):
        # question = "What is the brown table surrounded by?"
        # phrases = ["brown padded chairs", "by brown chairs"]
        gt_answer, pred_answer = qap_pairs[qid] # while we once noted the first item as questions, it is actually the ref_captions list, so it's the answer.
        gt_answer = "\n".join(f'"{p}"' for p in gt_answer) + "."

        content = GRADER_TEMPLATE.format(
            ground_truth=gt_answer,
            predicted_answer=pred_answer
        )

        response = client.responses.create(
            model="gpt-5-mini",
            reasoning={"effort": "minimal"},
            instructions="Act like a professional grader that always follows instructions and reasons carefully for the questions.",
            input=content,
        )

        answer = response.output_text

        if len(answer) != 1 or (answer != "A" and answer != "B"):
            print(f"Failed to output correct format answer for question id {qid}. \nanswer: {answer}", flush=True)
            # raise ValueError
            failed_ids.append(qid)
            judgements[qid] = False # just designate all ambiguous cases as false
        else:
            print(qid, answer, answer=="A", flush=True)
            judgements[qid] = answer == "A"

        ans_file.write(json.dumps({qid: judgements[qid]}) + "\n")
        ans_file.flush()
    ans_file.close()

    print(f"final accuracy: {sum(judgements.values()) / len(judgements) if len(judgements) > 0 else 0:.5f}", flush=True)

    if len(failed_ids) > 0:
        with open(output_failed_ids, "w") as f:
            for fid in failed_ids:
                f.write(f"{fid}\n")
        print(f"Some question ids failed to be classified. See {output_failed_ids} for details.", flush=True)
