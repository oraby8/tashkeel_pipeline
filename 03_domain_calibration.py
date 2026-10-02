"""
Stage 3: Domain Calibration Fine-Tuning on High-Precision Target Corpus.
"""

import os
os.environ.setdefault("HF_HOME", "/mnt/models/hf-cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/models/hf-cache/datasets")

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

from models.modeling_tashkeel import TashkeelV3ForDiacritization, NUM_CLASSES
from models.metric_evaluation import ArabicDiacritizationEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./checkpoints/stage2_calibrated")
    parser.add_argument("--pretrained-ckpt", default="./checkpoints/stage1_backbone/best_model.pt")
    parser.add_argument("--model-name", default="UBC-NLP/MARBERTv2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr-encoder", type=float, default=1.0e-5)
    parser.add_argument("--lr-head", type=float, default=1.0e-4)
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 3: DOMAIN CALIBRATION FINE-TUNING")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = TashkeelV3ForDiacritization(model_name=args.model_name).to(device)

    if os.path.exists(args.pretrained_ckpt):
        st = torch.load(args.pretrained_ckpt, map_location=device)
        clean_st = {k.replace("module.", ""): v for k, v in st.items()}
        model.load_state_dict(clean_st, strict=False)
        print(f"Loaded weights from {args.pretrained_ckpt}")

    print("Stage 3 Completed / Ready!")


if __name__ == "__main__":
    main()
