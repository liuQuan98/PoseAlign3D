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
Your job is to look at a question asking about the interior of a room and a bunch of associated ground truth answers, and then assign the problem as either [A: "NEED_LATERAL_DIRECTION", B: "DO_NOT_NEED_LATERAL_DIRECTION"] in order to answer.
First, I will give examples of each class, and then you will classify a new question.


The following are examples of DO_NOT_NEED_LATERAL_DIRECTION questions.
```
Question 1: What color is the chair in the kitchen?
Answer 1: "brown", "dark brown".
Question 2: What is the top of the table?
Answer 2: "tv", "television".
Question 3: What is placed next to the fridge?
Answer 3: "door", "the beige door".
Question 4: Where are the two chairs in front of one another?
Answer 4: "end table closest to door", "across from each other over table"
```
These questions all DO_NOT_NEED_LATERAL_DIRECTION because:
    - Either they ask a different trait of the object than its direction (e.g., color, material, shape, size, etc.).
    - Or the direction is dependent on an absolute direction (e.g., "top", "bottom", "next to", "between", "end") which can be implied without additional direction information.
    - Or like Question 4 where the reference frame is not needed or can be implied despite the existence of directional words like "in front of", because two chairs facing each other will be facing each other no matter where you look at them from.
    - While the question may contain spatial words, it does not necessarily need the directional information to answer.


The following are examples of question that NEED_LATERAL_DIRECTION.
```
Question 1: How many brown chairs are on the left of the brown table?
Answer 1: "4", "4"
Question 2: Where is the tall chair on the right of the table?
Answer 2: "to left of narrow table tv", "next to 2 other chairs as third chair on right"
Question 3: What part of the room has a fridge to its right?
Answer 3: "right corner wall adjacent to tv", "corner with tv alongside same wall"
Question 4: Where is the beige wooden desk placed?
Answer 4: "up against wall", "at front of class"
Question 5: What is on the front of the brown table?
Answer 5: "tv", "chair"
```
These predicted answers all NEED_LATERAL_DIRECTION because:
    - A relative direction word (e.g., "left", "right", "front", "back", "against") is present in the question or any of the answers.
    - Even if the question asks about directions relative to an object, in some cases the reference frame still cannot be implied, e.g., cylindrical or symmetric objects like dishes, tables, beds, etc, just like Question 5 above.
    - The question may be asking about a different trait of the object than its direction (e.g., color, material, shape, size, etc.), but the question cannot be answered without the directional information.
    
Also note the following things:
- For classifying the questions where the answers or question may seem odd, ignore that and just focus on whether the question can be answered without directional information.
- The answers may not coincide with each other, but that is okay. There can be multiple objects to the left or right of something. Just focus on whether the question can be answered without directional information.

Here is a new question. Simply reply with either NEED_LATERAL_DIRECTION or DO_NOT_NEED_LATERAL_DIRECTION. Do not ask any more questions to the user.
```
Question: {question}
Answers: {gt_answer}
```

Classify the question as one of:
A: NEED_LATERAL_DIRECTION
B: DO_NOT_NEED_LATERAL_DIRECTION

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
    parser.add_argument("--folder-name", type=str, default="./llm_as_judge/scanqa_vllm")
    args = parser.parse_args()

    basepath = args.folder_name
    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    output_file = os.path.join(basepath, f"{args.num_chunks}_{args.chunk_idx}.jsonl")
    output_failed_ids = os.path.join(basepath, f"{args.num_chunks}_{args.chunk_idx}_failed_ids.txt")

 
    # --- Reuse vLLM ---
    llm = LLM(
        model="openai/gpt-oss-20b",
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=8192,
        max_model_len=8200,
        tensor_parallel_size=1
    )

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    sampling = SamplingParams(
        max_tokens=8192,
        temperature=1,
        stop_token_ids=stop_token_ids,
    )

    question_file = "playground/data/eval_info/scanqa/scanqa_val_question.jsonl"
    answer_file = "playground/data/eval_info/scanqa/scanqa_val_answer.jsonl"

    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    answers = [json.loads(a) for a in open(os.path.expanduser(answer_file), "r")]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_map = {a["question_id"]: a['text'] for a in answers}
    qa_pairs = {q["question_id"]: [q['text'], answers_map[q["question_id"]]] for q in questions}

    judgements = {}
    failed_ids = []

    ans_file = open(output_file, "w")

    for qid in tqdm(qa_pairs.keys()):
        # question = "What is the brown table surrounded by?"
        # phrases = ["brown padded chairs", "by brown chairs"]
        question, phrases = qa_pairs[qid]
        result = ", ".join(f'"{p}"' for p in phrases) + "."

        # --- 1) get the input ---
        content = GRADER_TEMPLATE.format(
            question=question,
            gt_answer=result,
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
            # raise ValueError
            failed_ids.append(qid)
            judgements[qid] = False # just designate all ambiguous cases as false
        else:
            print(qid, answer, answer=="A", flush=True)
            judgements[qid] = answer == "A"

        ans_file.write(json.dumps({qid: judgements[qid]}) + "\n")
        ans_file.flush()
    ans_file.close()

    if len(failed_ids) > 0:
        with open(output_failed_ids, "w") as f:
            for fid in failed_ids:
                f.write(f"{fid}\n")
        print(f"Some question ids failed to be classified. See {output_failed_ids} for details.", flush=True)
