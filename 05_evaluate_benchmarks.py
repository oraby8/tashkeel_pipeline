"""
Stage 5: Official Benchmark Evaluation on SadeedDiac-25 and CATT_benchmark.
"""

import os
os.environ.setdefault("HF_HOME", "/mnt/models/hf-cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/models/hf-cache/datasets")

import argparse
import json
import warnings
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from models.modeling_tashkeel import TashkeelV3ForDiacritization, NUM_CLASSES
from models.metric_evaluation import ArabicDiacritizationEvaluator


def evaluate_dataset(model, tokenizer, dataset_id, split, device, batch_size=64):
    print(f"\nEvaluating on {dataset_id} ({split})...")
    ds = load_dataset(dataset_id, split=split)
    inputs = list(ds["input"])
    gts = list(ds["output"])
    print(f"Loaded {len(ds)} test sentences.")

    preds = model.diacritize(inputs, tokenizer=tokenizer, device=device, batch_size=batch_size, use_viterbi=True)

    scored = 0
    for p, g in zip(preds, gts):
        try:
            ArabicDiacritizationEvaluator.caculate_error_on_single_sentence(p.strip(), g.strip(), False)
            scored += 1
        except RuntimeError:
            pass

    hallucination_rate = 100.0 - (scored / len(ds) * 100.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Total_WER, Morph_WER, Total_DER, Morph_DER, NVW = ArabicDiacritizationEvaluator.caculate_errors_on_sentences(
            preds, gts, False
        )

    return {
        "dataset": dataset_id,
        "scored": scored,
        "total": len(ds),
        "hallucination_rate_pct": hallucination_rate,
        "Total_DER_CE": Total_DER,
        "Morph_DER_no_CE": Morph_DER,
        "Total_WER_CE": Total_WER,
        "Morph_WER_no_CE": Morph_WER,
        "NVW": NVW,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/models/tashkeel/tashkeel_v3_hf_upload")
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("STAGE 5: OFFICIAL BENCHMARK EVALUATION (VITERBI DECODING)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = TashkeelV3ForDiacritization(model_name="UBC-NLP/MARBERTv2").to(device)

    if args.ckpt_path:
        st = torch.load(args.ckpt_path, map_location=device)
        clean_st = {k.replace("module.", ""): v for k, v in st.items()}
        model.load_state_dict(clean_st)
    else:
        # Load directly from export directory
        weights_file = os.path.join(args.model_path, "pytorch_model.bin")
        if os.path.exists(weights_file):
            st = torch.load(weights_file, map_location=device)
            model.load_state_dict(st)

    model.eval()

    r_sadeed = evaluate_dataset(model, tokenizer, "Misraj/SadeedDiac-25", "train", device, batch_size=args.batch_size)
    r_catt = evaluate_dataset(model, tokenizer, "Bisher/CATT_benchmark", "train", device, batch_size=args.batch_size)

    print("\n" + "=" * 80)
    print("BENCHMARK REPORT SUMMARY")
    print("=" * 80)
    print(f"{'Benchmark':<30} | {'Total DER':<10} | {'Morph DER':<10} | {'Total WER':<10} | {'Morph WER':<10}")
    print("-" * 80)
    for r in [r_sadeed, r_catt]:
        print(f"{r['dataset']:<30} | {r['Total_DER_CE']:8.4f}% | {r['Morph_DER_no_CE']:8.4f}% | {r['Total_WER_CE']:8.4f}% | {r['Morph_WER_no_CE']:8.4f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
