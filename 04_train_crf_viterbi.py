"""
Stage 4: Linear-Chain CRF Training with Viterbi Sequence Optimization.
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

from models.modeling_tashkeel import TashkeelV3ForDiacritization, NUM_CLASSES, DIACRITICS_LIST, segment_word_morphemes
from models.metric_evaluation import ArabicDiacritizationEvaluator
from train_char_tagger_full import extract_char_diac_pairs


class CRFDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_crf_batch(batch, tokenizer, max_char_len=384, max_tok_len=192):
    valid = []
    for s in batch:
        inp = s[0] if isinstance(s, (list, tuple)) else s.get("input")
        out = s[1] if isinstance(s, (list, tuple)) else s.get("output")
        if isinstance(inp, str) and isinstance(out, str):
            c_labels = extract_char_diac_pairs(inp, out)
            if len(c_labels) == len(inp) and 0 < len(inp) <= max_char_len:
                valid.append((inp, out, c_labels))

    if not valid:
        return None

    texts = [v[0] for v in valid]
    enc = tokenizer(
        texts,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=max_tok_len,
        return_tensors="pt",
    )

    B = len(valid)
    max_c_len = max(len(t) for t in texts)
    num_tokens = enc["input_ids"].shape[1]
    offsets_batch = enc["offset_mapping"]
    if hasattr(offsets_batch, "tolist"):
        offsets_batch = offsets_batch.tolist()

    char_to_token = torch.zeros((B, max_c_len), dtype=torch.long)
    char_mask = torch.zeros((B, max_c_len), dtype=torch.bool)
    char_ids = torch.zeros((B, max_c_len), dtype=torch.long)
    word_pos_ids = torch.zeros((B, max_c_len), dtype=torch.long)
    morpheme_type_ids = torch.zeros((B, max_c_len), dtype=torch.long)
    is_irab_pos = torch.zeros((B, max_c_len), dtype=torch.bool)

    targets = torch.zeros((B, max_c_len), dtype=torch.long)
    ce_targets = torch.full((B, max_c_len), -100, dtype=torch.long)
    irab_targets = torch.full((B, max_c_len), -100, dtype=torch.long)

    for b in range(B):
        text, _, labels = valid[b]
        offsets = offsets_batch[b]
        tok_idx = 0
        words = text.split(" ")
        char_in_word_map, morpheme_map, is_stem_end_map = [], [], []

        for w in words:
            wl = len(w)
            spans = segment_word_morphemes(w)
            word_morpheme_types = [0] * wl
            word_stem_ends = [False] * wl

            for start, end, chunk, m_type in spans:
                for idx_c in range(start, end):
                    word_morpheme_types[idx_c] = m_type
                if m_type == 2 and end > start:
                    word_stem_ends[end - 1] = True

            for c_i in range(wl):
                char_in_word_map.append(min(c_i, 31))
                morpheme_map.append(word_morpheme_types[c_i])
                is_stem_end_map.append(word_stem_ends[c_i] or c_i == wl - 1)

            char_in_word_map.append(0)
            morpheme_map.append(0)
            is_stem_end_map.append(False)

        for c_idx in range(len(text)):
            while tok_idx < len(offsets) and offsets[tok_idx][1] <= c_idx:
                tok_idx += 1
            char_to_token[b, c_idx] = min(tok_idx, num_tokens - 1)
            char_mask[b, c_idx] = True
            targets[b, c_idx] = labels[c_idx]
            ce_targets[b, c_idx] = labels[c_idx]
            char_ids[b, c_idx] = min(ord(text[c_idx]), 255)

            if c_idx < len(char_in_word_map):
                word_pos_ids[b, c_idx] = char_in_word_map[c_idx]
                morpheme_type_ids[b, c_idx] = morpheme_map[c_idx]
                is_end = is_stem_end_map[c_idx]
                is_irab_pos[b, c_idx] = is_end

                if is_end and ArabicDiacritizationEvaluator.has_arabic_letter(text[c_idx]):
                    irab_targets[b, c_idx] = labels[c_idx]

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "char_to_token": char_to_token,
        "char_mask": char_mask,
        "char_ids": char_ids,
        "word_pos_ids": word_pos_ids,
        "morpheme_type_ids": morpheme_type_ids,
        "is_irab_pos": is_irab_pos,
        "targets": targets,
        "ce_targets": ce_targets,
        "irab_targets": irab_targets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./checkpoints/stage3_crf_final")
    parser.add_argument("--pretrained-ckpt", default="./checkpoints/e018-crf-tagger/best_model.pt")
    parser.add_argument("--data-path", default="/mnt/models/hf-cache/sadeed_silver_pos_300k.json")
    parser.add_argument("--model-name", default="UBC-NLP/MARBERTv2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr-encoder", type=float, default=1.8e-5)
    parser.add_argument("--lr-head", type=float, default=2.0e-4)
    args = parser.parse_args()

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank, global_rank, world_size = 0, 0, 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 70)
        print(f"STAGE 4: LINEAR-CHAIN CRF TRAINING WITH VITERBI DECODING [DDP World Size: {world_size}]")
        print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = TashkeelV3ForDiacritization(model_name=args.model_name).to(device)

    if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
        st = torch.load(args.pretrained_ckpt, map_location=device)
        clean_st = {k.replace("module.", ""): v for k, v in st.items()}
        base_model.load_state_dict(clean_st, strict=False)
        if global_rank == 0:
            print(f"Loaded weights from {args.pretrained_ckpt}")

    if is_distributed:
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = base_model

    with open(args.data_path, "r", encoding="utf-8") as f:
        data_blob = json.load(f)
    samples = data_blob["samples"] if isinstance(data_blob, dict) and "samples" in data_blob else data_blob

    train_dataset = CRFDataset(samples)
    sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // world_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_crf_batch(b, tokenizer),
    )

    optimizer_grouped = [
        {"params": [p for n, p in base_model.encoder.named_parameters() if p.requires_grad], "lr": args.lr_encoder},
        {"params": [p for n, p in base_model.named_parameters() if not n.startswith("encoder.") and p.requires_grad], "lr": args.lr_head},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)

    for epoch in range(args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}") if global_rank == 0 else train_loader

        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(
                batch_gpu["input_ids"],
                batch_gpu["attention_mask"],
                batch_gpu["char_to_token"],
                batch_gpu["char_mask"],
                char_ids=batch_gpu["char_ids"],
                word_pos_ids=batch_gpu["word_pos_ids"],
                morpheme_type_ids=batch_gpu["morpheme_type_ids"],
                is_irab_pos=batch_gpu["is_irab_pos"],
            )

            emissions = outputs["emissions"]
            stem_logits = outputs["stem_logits"]
            irab_logits = outputs["irab_logits"]
            targets = batch_gpu["targets"]
            ce_targets = batch_gpu["ce_targets"]
            irab_targets = batch_gpu["irab_targets"]
            char_mask = batch_gpu["char_mask"]

            crf_module = base_model.crf
            loss_crf = crf_module(emissions, targets, char_mask)
            l_stem = F.cross_entropy(stem_logits.view(-1, NUM_CLASSES), ce_targets.view(-1), ignore_index=-100, label_smoothing=0.03)
            irab_mask = irab_targets.view(-1) != -100
            l_irab = F.cross_entropy(irab_logits.view(-1, NUM_CLASSES)[irab_mask], irab_targets.view(-1)[irab_mask], label_smoothing=0.03) if irab_mask.sum() > 0 else torch.tensor(0.0, device=device)

            loss = loss_crf + 0.5 * l_stem + 1.5 * l_irab
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if global_rank == 0:
            torch.save(base_model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    if global_rank == 0:
        torch.save(base_model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        print("Stage 4 Training Complete!")


if __name__ == "__main__":
    main()
