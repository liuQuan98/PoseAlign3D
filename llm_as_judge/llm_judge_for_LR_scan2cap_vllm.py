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
Your job is to look at a bunch of ground truth descriptions of a single object in a room for a referral caption task, and then assign the problem as either [A: "NEED_LATERAL_DIRECTION", B: "DO_NOT_NEED_LATERAL_DIRECTION"] in order to answer.
First, I will give examples of each class, and then you will classify a new question.


The following is an example of DO_NOT_NEED_LATERAL_DIRECTION questions.
```
Answers:
    "There is a rectangular cabinet overhanging. It is next to a copier.",
    "A broad black closed cabinet. It is attached to the wall.",
    "A black broad cabinet. It is attached to the wall.",
    "The cabinet is affixed to the wall. It is above the printer and the lower counter.",
    "The cabinet is located on the left side of the room. It is affixed to the wall. It is above the printer below it."
```
This question is DO_NOT_NEED_LATERAL_DIRECTION because:
    - Either they describe the location relative to the room (e.g., "left side of the room",) rather than relative to another object.
    - Or the direction is not dependent on a lateral direction (e.g., "above", "next to", "over", etc.).
    - Or the description is about other traits of the object than its direction (e.g., color, material, shape, size, etc.).
    - Or the description do not contain any directional information at all.


The following are examples of question that NEED_LATERAL_DIRECTION.
```
Answers:
    "The window has a white frame. It is located behind the brown dresser, at its left end.",
    "It is a blue window. The blue window is sitting behind the front wall right next to the brown cabinet.",
    "The window is on the left side of the room. It is to the left of the cabinet and the trash can.",
    "This is a glass window. It is to the left of another window.",
    "A clear compact window. It is located near the machine."
```
This question is NEED_LATERAL_DIRECTION because:
    - A relative direction word (e.g., "left", "right", "front", "back", "against") is present and describes relative position against some other object.
    - The description may contain different traits of the object than its direction (e.g., color, material, shape, size, etc.), but the direction words constitute a significant part in localizing it, which cannot be complete without the directional information.
    
Also note the following things:
- For the questions where the answers may seem odd, ignore that and just focus on whether the question can be answered without directional information.
- The answers may not coincide with each other, but that is okay. There can be multiple objects to the left or right of something. Just focus on whether the question can be answered without directional information.

Here is a new question. Simply reply with either NEED_LATERAL_DIRECTION or DO_NOT_NEED_LATERAL_DIRECTION. Do not ask any more questions to the user.
```
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
    parser.add_argument("--folder-name", type=str, default="./llm_as_judge/scan2cap_vllm_corpus")
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

    answer_file = "playground/data/eval_info/densecap_scanrefer/scan2cap_val_corpus.json"
    answers = json.load(open(answer_file, "r"))
    keys_list = [key for key in answers.keys()]
    keys_list = get_chunk(keys_list, args.num_chunks, args.chunk_idx)

    qa_pairs = {q: answers[q] for q in keys_list}

    judgements = {}
    failed_ids = []

    ans_file = open(output_file, "w")

    for qid in tqdm(qa_pairs.keys()):
        # question = "What is the brown table surrounded by?"
        # phrases = ["brown padded chairs", "by brown chairs"]
        phrases = qa_pairs[qid]
        text = "\n    ".join([f'"{p}"' for p in phrases])

        # --- 1) get the input ---
        content = GRADER_TEMPLATE.format(
            gt_answer=text,
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
