import json

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing LLM as judge results.",
    )

    args = parser.parse_args()

    llm_as_judge_results = [json.loads(q) for q in open(args.input_file, "r")]
    # filter questions to only those in the subset
    subset_llm_as_judge_results = llm_as_judge_results

    # get the question ids that LLM-as-judge decided as 'A' (correct)
    llm_as_judge_hits = [[key for key, item in subset_llm_as_judge_results[idx].items() if item] for idx in range(len(subset_llm_as_judge_results))]
    llm_as_judge_hits = [item[0] for item in llm_as_judge_hits if len(item) > 0]

    val_scores = {}
    val_scores["LLM-as-judge Acc"] = len(llm_as_judge_hits) / len(subset_llm_as_judge_results) if len(subset_llm_as_judge_results) > 0 else 0
    print(val_scores)
