import torch
import json
from tqdm import tqdm
import math
import argparse


import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
 
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
import os


GRADER_TEMPLATE = """
Your job is to look at a question, a set of ground-truth answers, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question 1: I was tired so I saw in the armchair with the doors to my right. What color is the armchair in front of me?
Ground truth: "brown"
Predicted answer 1: brown.
Predicted answer 2: dark brown.
Predicted answer 3: chocolate.
Predicted answer 4: hazel.

Question 2: I am sitting on the chair with my back to the window enjoying my cup of orange juice. What type of box to my right that are top of the stool containing tools?
Ground truth: "toolboxes"
Predicted answer 1: workbox.
Predicted answer 2: toolboxes.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in one of the ground truths. Aliases, plurals, and synonyms are acceptable.
    - They do not contain any information that contradicts the ground truth.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the ground truth is fully included and the response contains no incorrect information or contradictions.

    
The following are examples of INCORRECT predicted answers.
```
Question 1: I am entering the room facing the armchair with a printer on my right. What is the color of the cup?
Ground truth: "blue"
Predicted answer 1: white.
Predicted answer 2: gray.
Predicted answer 3: transparent.

Question 2: I am sitting on piano bench facing the couch. How many chairs are on the right?
Ground truth: "one"
Predicted answer 1: one dozen.
Predicted answer 2: two.
Predicted answer 3: to right of refrigerator.
```
These predicted answers are all INCORRECT because:
    - None of the information in the ground truth is included in the answer.
    - Or the predicted answer contradicts facts contained in the question and the ground truth. Pay extra attention to semantic meaning than string matching. Having the correct words is not enough; the meaning must be correct.
    - The answer shies away from the question and does not provide any useful information.


Also note the following things:
- For grading questions where the ground truth is an estimated number rather than a counting number, for example, "How much larger is the brown cabinet compared to the purple stool?" with ground truth "5 times larger" should be graded as follows:
    - Predicted answers "5", "4", and "6" are all CORRECT, as the number being asked is an estimation rather than a precise count, and the predicted answer is close enough to the ground truth.
    - Predicted answers "2", "3" and "8" are INCORRECT, as they are not close enough to the ground truth.
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

    os.makedirs(os.path.join(basepath, "llm_as_judge"), exist_ok=True)
    output_file = os.path.join(basepath, "llm_as_judge", f"llm_as_judge_{args.num_chunks}_{args.chunk_idx}.jsonl")
    output_failed_ids = os.path.join(basepath, "llm_as_judge", f"llm_as_judge_{args.num_chunks}_{args.chunk_idx}_failed_ids.txt")
 
    # --- 1) Reuse vLLM ---
    llm = LLM(
        model="openai/gpt-oss-20b",
        trust_remote_code=True,
        gpu_memory_utilization = 0.8,
        max_num_batched_tokens=4096,
        max_model_len=4100,
        tensor_parallel_size=1
    )

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    sampling = SamplingParams(
        max_tokens=8192,
        temperature=1,
        stop_token_ids=stop_token_ids,
    )

    question_file = "playground/data/eval_info/sqa3d/sqa3d_test_question.json"
    answer_file = "playground/data/eval_info/sqa3d/sqa3d_test_answer.json"
    prediction_file = os.path.join(args.prediction_file_path, "merge.jsonl")

    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    answers = [json.loads(a) for a in open(os.path.expanduser(answer_file), "r")]
    predicted_answers = [json.loads(p) for p in open(os.path.expanduser(prediction_file), "r")]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    purge_sentence_end = " Answer the question using a single word or phrase."

    answers_map = {a["question_id"]: a['text'] for a in answers}
    predicted_answers_map = {p["question_id"]: p['text'] for p in predicted_answers}
    qap_pairs = {q["question_id"]: [q['text'][:-len(purge_sentence_end)], answers_map[q["question_id"]], predicted_answers_map[q["question_id"]]] for q in questions}

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

        # --- 2) Render the prefill with Harmony ---
        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                Message.from_role_and_content(Role.USER, content),
            ]
        )
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

        # --- 3) Run vLLM with prefill ids. The input ids can be still squeezed in under vllm=0.10.2 using the TokensPrompt class ---
        outputs = llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=prefill_ids)],
            sampling_params=sampling,
            use_tqdm=False,
        )
        
        gen = outputs[0].outputs[0]
        output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)

        # --- 4) Parse the completion token IDs back into structured Harmony messages ---
        try:
            entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
        except Exception as e:
            entries = []
            print(f"Failed to parse messages from completion tokens for question id {qid} due to {e}. \nFull content: {content}", flush=True)

        # 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
        answer = ""
        analysis = ""
        for message in entries:
            response = message.to_dict()
            if response["role"] == "assistant" and response["channel"] == "final":
                answer = response["content"][0]["text"].strip()
            if response["role"] == "assistant" and response["channel"] == "analysis":
                analysis = response["content"][0]["text"].strip()

        if len(answer) != 1 or (answer != "A" and answer != "B"):
            print(f"Failed to find cut position in answer for question id {qid}. \nFull analysis: {analysis}, \nanswer: {answer}", flush=True)
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
