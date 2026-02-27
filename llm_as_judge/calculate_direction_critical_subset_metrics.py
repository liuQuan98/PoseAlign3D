import re
import json
import argparse
import copy
import os
import torch
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from llava.eval.box_utils import box3d_iou, construct_bbox_corners


tokenizer = PTBTokenizer()
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    # (Spice(), "SPICE")
]


# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/data/data_utils.py#L322-L379
def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/evaluator/scanqa_eval.py#L41-L50
def answer_match(pred, gts):
    # return EM and refined EM
    if pred in gts:
        return 1, 1
    for gt in gts:
        if ''.join(pred.split()) in ''.join(gt.split()) or ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0

def calc_scanqa_score(preds, gts, tokenizer, scorers):
    val_scores = {}
    tmp_preds = {}
    tmp_targets = {}
    acc, refined_acc = 0, 0
    print("Total samples:", len(preds))
    assert len(preds) == len(gts)  # number of samples
    for item_id, (pred, gt) in tqdm(enumerate(zip(preds, gts))):
        question_id = pred['question_id']
        gt_question_id = gt['question_id']
        assert question_id == gt_question_id
        pred_answer = pred['text']
        gt_answers = gt['text']
        pred_answer = clean_answer(pred_answer)
        ref_captions = [clean_answer(gt_answer) for gt_answer in gt_answers]
        tmp_acc, tmp_refined_acc = answer_match(pred_answer, ref_captions)
        acc += tmp_acc
        refined_acc += tmp_refined_acc
        tmp_preds[item_id] = [{'caption': pred_answer}]
        ref_captions = [p.replace("\n", " ").strip() for p in ref_captions]
        tmp_targets[item_id] = [{'caption': caption} for caption in ref_captions]
    tmp_preds = tokenizer.tokenize(tmp_preds)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    acc = acc / len(preds)
    refined_acc = refined_acc / len(preds)
    val_scores["[scanqa] EM1"] = acc
    val_scores["[scanqa] EM1_refined"] = refined_acc
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                val_scores[f"[scanqa] {m}"] = sc
        else:
            val_scores[f"[scanqa] {method}"] = score
    return val_scores


def calc_scan2cap_score(preds, tokenizer, scorers, args, gt_dict):
    instance_attribute_file = args.pred_instance_attribute_file
    scannet_attribute_file = args.gt_instance_attribute_file

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    # gt_dict = json.load(open('annotations/scan2cap_val_corpus.json'))
    tmp_preds_iou25 = {}
    tmp_preds_iou50 = {}
    tmp_targets = {}
    for pred in preds:
        scene_id = pred['scene_id']
        pred_id = pred['pred_id']
        gt_id = pred['gt_id']
        pred_locs = instance_attrs[scene_id]['locs'][pred_id].tolist()
        gt_locs = scannet_attrs[scene_id]['locs'][gt_id].tolist()
        pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
        gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
        iou = box3d_iou(pred_corners, gt_corners)
        key = f"{scene_id}|{gt_id}"
        if iou >= 0.25:
            tmp_preds_iou25[key] = [{'caption': f"sos {pred['text']} eos".replace('\n', ' ')}]
        else:
            tmp_preds_iou25[key] = [{'caption': f"sos eos"}]
        if iou >= 0.5:
            tmp_preds_iou50[key] = [{'caption': f"sos {pred['text']} eos".replace('\n', ' ')}]
        else:
            tmp_preds_iou50[key] = [{'caption': f"sos eos"}]
        tmp_targets[key] = [{'caption': caption} for caption in gt_dict[key]]
    
    missing_keys = gt_dict.keys() - tmp_targets.keys()

    for missing_key in missing_keys:
        tmp_preds_iou25[missing_key] = [{'caption': "sos eos"}]
        tmp_preds_iou50[missing_key] = [{'caption': "sos eos"}]
        tmp_targets[missing_key] = [{'caption': caption} for caption in gt_dict[missing_key]]
    
    tmp_preds_iou25 = tokenizer.tokenize(tmp_preds_iou25)
    tmp_preds_iou50 = tokenizer.tokenize(tmp_preds_iou50)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    val_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds_iou25)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                val_scores[f"[scan2cap] {m}@0.25"] = sc
        else:
            val_scores[f"[scan2cap] {method}@0.25"] = score
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds_iou50)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                val_scores[f"[scan2cap] {m}@0.50"] = sc
        else:
            val_scores[f"[scan2cap] {method}@0.50"] = score
    return val_scores


def main(args):
    if not args.llm_only:
        if 'scanqa' in args.annotation_file.lower():
            print("Evaluating on ScanQA dataset...")
            preds = [json.loads(q) for q in open(args.result_file, "r")]
            questions = [json.loads(q) for q in open(args.question_file, "r")]
            gts   = [json.loads(q) for q in open(args.annotation_file, "r")]

            if args.use_llm_decided_subset:
                subset_ids = [json.loads(q) for q in open(args.subset_file, "r")]
                subset_ids = [[key for key, item in subset_ids[idx].items() if item] for idx in range(len(subset_ids))]
                subset_ids = [item[0] for item in subset_ids if len(item) > 0]
            else:
                subset_ids = [q["question_id"] for q in questions if 'left' in q["text"].lower() or 'right' in q["text"].lower()]

            print(f"Total {len(preds)} predictions, {len(gts)} ground truths, {len(subset_ids)} subset questions, subset portion {len(subset_ids) / len(questions) if questions else 0:.5f}.")

            if args.use_complementary_subset is None:
                pass # use all data
            elif args.use_complementary_subset: # use outlier subset
                preds = [item for item in preds if item['question_id'] not in subset_ids]
                gts   = [item for item in gts if item['question_id'] not in subset_ids]
            else: # use inlier subset
                preds = [item for item in preds if item['question_id'] in subset_ids]
                gts   = [item for item in gts if item['question_id'] in subset_ids]
            val_scores = calc_scanqa_score(preds, gts, tokenizer, scorers)
        elif 'densecap_scanrefer' in args.annotation_file.lower():
            print("Evaluating on Scan2Cap dataset...")
            preds = [json.loads(q) for q in open(args.result_file, "r")]

            if args.use_llm_decided_subset:
                subset_ids_all = [json.loads(q) for q in open(args.subset_file, "r")]
                subset_ids = [[key for key, item in subset_ids_all[idx].items() if item] for idx in range(len(subset_ids_all))]
                subset_ids = [item[0] for item in subset_ids if len(item) > 0]
            else:
                gt_dict = json.load(open(args.annotation_file))
                gt_concatenated = {key: ' '.join(gt_dict[key]) for key in gt_dict.keys()}
                subset_ids_all = list(gt_concatenated.keys()) # for portion calculation only
                subset_ids = [key for key in gt_concatenated.keys() if 'left' in gt_concatenated[key].lower() or 'right' in gt_concatenated[key].lower()]

            print(subset_ids[:10])
            print(f"Total {len(preds)} predictions, {len(subset_ids)} subset questions, subset portion {len(subset_ids) / len(subset_ids_all) if subset_ids_all else 0:.5f}.")

            gt_dict = json.load(open(args.annotation_file))
            if args.use_complementary_subset is None:
                pass # use all data
            elif args.use_complementary_subset: # use outlier subset
                preds = [q for q in preds if f"{q['scene_id']}|{q['gt_id']}" not in subset_ids]
                gt_dict = {key: gt_dict[key] for key in gt_dict.keys() if key not in subset_ids}
            else: # use inlier subset
                preds = [q for q in preds if f"{q['scene_id']}|{q['gt_id']}" in subset_ids]
                gt_dict = {key: gt_dict[key] for key in gt_dict.keys() if key in subset_ids}

            val_scores = calc_scan2cap_score(preds, tokenizer, scorers, args, gt_dict)
    else:
        subset_ids = None
        val_scores = {}

    if args.llm_as_judge_file is not None and os.path.exists(args.llm_as_judge_file):
        # load the LLM-as-judge results
        llm_as_judge_results = [json.loads(q) for q in open(args.llm_as_judge_file, "r")]
        if 'densecap_scanrefer' in args.annotation_file.lower():
            key_aligned_results = []
            for item in llm_as_judge_results:
                key = list(item.keys())[0]
                key_scene_gt_pred = key.replace("_obj-id_", "|")
                pred_location = key_scene_gt_pred.rfind("_pred-id_")
                key_scene_gt = key_scene_gt_pred[:pred_location]
                key_aligned_results.append({key_scene_gt: list(item.values())[0]})
                # substitute the keys to match those in playground/data/eval_info/densecap_scanrefer/scan2cap_val_corpus.json
            llm_as_judge_results = key_aligned_results
        # filter questions to only those in the subset
        if args.use_complementary_subset is None:
            subset_llm_as_judge_results = llm_as_judge_results
        elif args.use_complementary_subset:
            subset_llm_as_judge_results = [item for item in llm_as_judge_results if list(item.keys())[0] not in subset_ids]
        else:
            subset_llm_as_judge_results = [item for item in llm_as_judge_results if list(item.keys())[0] in subset_ids]
        # get the question ids that LLM-as-judge decided as 'A' (correct)
        llm_as_judge_hits = [[key for key, item in subset_llm_as_judge_results[idx].items() if item] for idx in range(len(subset_llm_as_judge_results))]
        llm_as_judge_hits = [item[0] for item in llm_as_judge_hits if len(item) > 0]

        print(f"Total {len(llm_as_judge_results)} questions available after filter, {len(subset_llm_as_judge_results)} subset questions calculated, {len(llm_as_judge_hits)} hits.")
        val_scores["LLM-as-judge Acc"] = len(llm_as_judge_hits) / len(subset_llm_as_judge_results) if len(subset_llm_as_judge_results) > 0 else 0
    print(val_scores)

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default=None)
    parser.add_argument('--question-file', type=str, default=None)
    parser.add_argument('--result-file', type=str, default=None)
    parser.add_argument('--llm-as-judge-file', type=str, default=None)
    parser.add_argument('--subset-file', type=str, default=None)
    parser.add_argument('--use-llm-decided-subset', type=str2bool, default=False)
    parser.add_argument('--use-complementary-subset', type=str2bool, default=False)
    parser.add_argument('--llm-only', type=str2bool, default=False)
    parser.add_argument('--pred-instance-attribute-file', type=str, default="")
    parser.add_argument('--gt-instance-attribute-file', type=str, default="")
    args = parser.parse_args()

    if args.use_complementary_subset:
        args_copy_inlier = copy.deepcopy(args)
        args_copy_outlier = copy.deepcopy(args)
        args_copy_inlier.use_complementary_subset = False
        args_copy_outlier.use_complementary_subset = True
        print("-------------------Inlier subset results:-------------------")
        main(args_copy_inlier)
        print("-------------------Outlier subset results:-------------------")
        main(args_copy_outlier)
    else:
        args.use_complementary_subset = None
        main(args)
