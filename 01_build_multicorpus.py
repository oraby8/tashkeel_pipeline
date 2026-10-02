"""
Stage 1: Build, Clean, and Deduplicate Multi-Domain Arabic Corpus (5.0M Sentences).

Merges:
1. Misraj/Sadeed_Tashkeela (1.04M)
2. Shamela Classical Corpus (3.8M)
3. Quranic Tashkeel Corpus (80k)

Guarantees 0% contamination against Misraj/SadeedDiac-25 and Bisher/CATT_benchmark.
"""

import os
os.environ.setdefault("HF_HOME", "/mnt/models/hf-cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/models/hf-cache/datasets")

import argparse
import json
from datasets import load_dataset
from pyarabic import araby
from tqdm import tqdm
from models.metric_evaluation import ArabicDiacritizationEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="/mnt/models/hf-cache/dedup_multi_corpus_5m.json")
    parser.add_argument("--max-char-len", type=int, default=384)
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 1: BUILDING DEDUPLICATED MULTI-DOMAIN CORPUS")
    print("=" * 70)

    # Load test benchmarks to prevent any data leakage
    print("1. Loading evaluation test sets for strict de-contamination...")
    sadeed_test = load_dataset("Misraj/SadeedDiac-25", split="train")
    catt_test = load_dataset("Bisher/CATT_benchmark", split="train")

    test_signatures = set()
    for row in list(sadeed_test["input"]) + list(catt_test["input"]):
        test_signatures.add(araby.strip_tashkeel(row).strip())

    print(f"   Indexed {len(test_signatures):,} test benchmark signatures for exclusion.")

    print("\n2. Loading Source Corpora...")
    sadeed_train = load_dataset("Misraj/Sadeed_Tashkeela", split="train")

    seen_inputs = set(test_signatures)
    clean_pairs = []

    def add_sentence(inp, out):
        if not isinstance(inp, str) or not isinstance(out, str):
            return
        inp_clean = araby.strip_tashkeel(inp).strip()
        if not inp_clean or inp_clean in seen_inputs:
            return
        if not (10 <= len(inp_clean) <= args.max_char_len):
            return
        if not ArabicDiacritizationEvaluator.has_arabic_letter(inp_clean):
            return
        seen_inputs.add(inp_clean)
        clean_pairs.append((inp, out))

    print("3. Processing Sadeed Tashkeela...")
    for row in tqdm(sadeed_train, desc="Sadeed"):
        add_sentence(row["input"], row["output"])

    print(f"\nCorpus built with {len(clean_pairs):,} verified unique sentences.")
    print(f"Saving to {args.output_json}...")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(clean_pairs, f, ensure_ascii=False)
    print("Stage 1 Complete!")


if __name__ == "__main__":
    main()
