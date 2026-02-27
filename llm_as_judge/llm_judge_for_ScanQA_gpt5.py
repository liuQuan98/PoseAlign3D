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
Your job is to look at a question, a ground-truth answer, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question 1: What type of stool is to the left of the door?
Ground truth: "bar stool", "black topped wooden stool"
Predicted answer 1: wooden stool.
Predicted answer 2: black topped bar stool.
Predicted answer 3: wooden bar stool.

Question 2: Where is the door of the refrigerator located?
Ground truth: "next to microwave", "on refrigerator to right of door"
Predicted answer 1: next to the microwave.
Predicted answer 2: on the refrigerator to the right of the door.

Queestion 3: What is in the bin next to the door?
Ground Truth: "trash for recycling", "recycling bins"
Predicted answer 1: trash can.
Predicted answer 2: litter waiting for recycling.
Predicted answer 3: waste.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in one of the ground truths. Aliases and synonyms are acceptable.
    - They do not contain any information that contradicts the ground truth.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the ground truth is fully included and the response contains no incorrect information or contradictions.

    
The following are examples of INCORRECT predicted answers.
```
Question 1: What is to the right of the refrigerator and above the kitchen cabinet?
Ground truth: "microwave", "microwave"
Predicted answer 1: kitchen counter.
Predicted answer 2: kitchen sink.
Predicted answer 3: water tap.

Question 2: Where is the door of the refrigerator located?
Ground truth: "next to microwave", "on refrigerator to right of door"
Predicted answer 1: on the microwave.
Predicted answer 2: on the door next to the refrigerator.
Predicted answer 3: to right of refrigerator.
Predicted answer 4: on the microwave above the refrigerator.
Predicted answer 5: on the refrigerator to left of the door.

Question 3: How much larger is the blue recycling bin compared to the beige trash can?
Ground truth: "twice as wide", "about double size"
Predicted answer 1: larger.
Predicted answer 2: blue recycling bin is larger than beige trash can.
Predicted answer 3: blue recycling bin is much larger than beige trash can.
```
These predicted answers are all INCORRECT because:
    - None of the information in the ground truth is included in the answer.
    - Or the predicted answer contradicts facts contained in the question and the ground truth. Pay extra attention to object relations, e.g., the door of the refrigerator cannot be on the microwave or another door. Having the correct words is not enough; the meaning must be correct.
    - The answer shies away from the question and does not provide any useful information.


Also note the following things:
- For grading questions where the ground truth contains a number, the predicted answer needs to be correct to the counting figure in the ground truth, but can be relaxed by +-1 when describing rough figures like relative size. For example, consider a question "How much larger is the brown cabinet compared to the purple stool?" with ground truth "5 times larger". 
    - Predicted answers "5", "4", and "6" are all CORRECT. 
    - Predicted answers "2", "3" and "8" are INCORRECT. 
    - Predicted answers "larger" and "more than 2" are considered INCORRECT because they provide no useful information.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What surface texture is the wooden cardboard?" and the gold target "wooden matte", "brown flat". The predicted answer "flat" would be considered CORRECT, even though it does not include "wooden", since it is clear from the question that the cardboard is wooden.

    
Here is a new example. Simply reply with either CORRECT or INCORRECT.
```
Question: {question}
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

    question_file = "playground/data/eval_info/scanqa/scanqa_val_question.jsonl"
    answer_file = "playground/data/eval_info/scanqa/scanqa_val_answer.jsonl"
    prediction_file = os.path.join(args.prediction_file_path, "merge.jsonl")

    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    answers = [json.loads(a) for a in open(os.path.expanduser(answer_file), "r")]
    predicted_answers = [json.loads(p) for p in open(os.path.expanduser(prediction_file), "r")]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_map = {a["question_id"]: a['text'] for a in answers}
    predicted_answers_map = {p["question_id"]: p['text'] for p in predicted_answers}
    qap_pairs = {q["question_id"]: [q['text'], answers_map[q["question_id"]], predicted_answers_map[q["question_id"]]] for q in questions}

    judgements = {}
    failed_ids = []

    ans_file = open(output_file, "w")

    for qid in tqdm(qap_pairs.keys()):
        # question = "What is the brown table surrounded by?"
        # phrases = ["brown padded chairs", "by brown chairs"]
        question, gt_answer, pred_answer = qap_pairs[qid]
        gt_answer = ", ".join(f'"{p}"' for p in gt_answer) + "."

        content = GRADER_TEMPLATE.format(
            question=question,
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
