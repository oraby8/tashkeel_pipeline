"""
Stage 6: Export Trained Checkpoint to Full Hugging Face Hub Package.
"""

import os
import shutil
import json
import torch
from transformers import AutoTokenizer
from safetensors.torch import save_file


def main():
    source_ckpt = "./checkpoints/stage3_crf_final/best_model.pt"
    if not os.path.exists(source_ckpt):
        source_ckpt = "/mnt/models/tashkeel/checkpoints/e018-crf-tagger/best_model.pt"

    output_dir = "./tashkeel_v3_export"
    os.makedirs(output_dir, exist_ok=True)
    base_model = "UBC-NLP/MARBERTv2"

    print("=" * 70)
    print("STAGE 6: EXPORTING HUGGING FACE PACKAGE")
    print("=" * 70)

    print("1. Saving Tokenizer...")
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.save_pretrained(output_dir)

    print(f"2. Converting Weights from {source_ckpt}...")
    st = torch.load(source_ckpt, map_location="cpu")
    clean_st = {k.replace("module.", ""): v.contiguous() for k, v in st.items()}

    torch.save(clean_st, os.path.join(output_dir, "pytorch_model.bin"))
    save_file(clean_st, os.path.join(output_dir, "model.safetensors"))
    print("   Saved pytorch_model.bin and model.safetensors!")

    print("3. Copying Model & Config Files...")
    shutil.copy("models/configuration_tashkeel.py", os.path.join(output_dir, "configuration_tashkeel.py"))
    shutil.copy("models/linear_chain_crf.py", os.path.join(output_dir, "linear_chain_crf.py"))
    shutil.copy("models/modeling_tashkeel.py", os.path.join(output_dir, "modeling_tashkeel.py"))

    config = {
        "architectures": ["TashkeelV3ForDiacritization"],
        "model_type": "tashkeel_v3",
        "auto_map": {
            "AutoConfig": "configuration_tashkeel.TashkeelConfig",
            "AutoModel": "modeling_tashkeel.TashkeelV3ForDiacritization"
        },
        "base_model_name_or_path": base_model,
        "num_classes": 15,
        "char_embed_dim": 64,
        "word_pos_embed_dim": 32,
        "morpheme_embed_dim": 16,
        "lstm_hidden_size": 384,
        "lstm_num_layers": 2,
        "vocab_size": len(tok),
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.45.0"
    }

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nPackage ready at: {output_dir}")
    print("Stage 6 Complete!")


if __name__ == "__main__":
    main()
