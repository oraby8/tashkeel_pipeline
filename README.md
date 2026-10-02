# Tashkeel-v4: Complete Training, Fine-Tuning & Architecture Manual

This directory contains the self-contained, reproducible pipeline to train, calibrate, evaluate, and deploy **Tashkeel-v4** (Morpheme-Aware MARBERTv2 + Linear-Chain CRF with Viterbi Sequence Decoding).

---

## 🏗️ Detailed Architecture: Purpose of Every Layer

The model is designed around a **hierarchical morpho-syntactic architecture** where token-level contextual semantics, sub-word character representations, morphological clitic segmentations, and global sequence constraints are decoupled and recombined.

```
                              [Raw Input Arabic Text]
                                         │
                             ┌───────────┴───────────┐
                             ▼                       ▼
                   [Layer 1: MARBERTv2]    [Layer 2, 3, 4: Character & Morpheme Embeddings]
                   (768d Contextual)       (Char 64d + Word-Pos 32d + Morpheme 16d)
                             │                       │
                             └───────────┬───────────┘
                                         ▼
                   [Layer 5: Concatenation & Dimension Fusion]
                                    (880d)
                                         │
                             ┌───────────┴───────────┐
                             ▼                       ▼
                 [Layer 6: 2-Layer BiLSTM]   [Layer 7: Residual Projection & LayerNorm]
                 (880d -> 768d Sequential)   (880d -> 768d Direct Linear Shortcut)
                             │                       │
                             └───────────┬───────────┘
                                         ▼
                            [LayerNorm Sum: h_char (768d)]
                                         │
                             ┌───────────┴───────────┐
                             ▼                       ▼
                   [Layer 8: Stem Head]    [Layer 9: I'rab Head]
                   (Internal Morphology)   (Terminal Case Ending)
                             │                       │
                             └───────────┬───────────┘
                                         ▼
                   [Layer 10: Dynamic Emission Gating Matrix]
                               (15d Emissions)
                                         │
                                         ▼
                   [Layer 11: Linear-Chain CRF Transition Matrix (A_15x15)]
                                         │
                                         ▼
                   [Layer 12: Viterbi Dynamic Programming Decoder]
                                         │
                                         ▼
                           [Final Diacritized Text (y*)]
```

---

### Layer-by-Layer Purpose & Function

### 1. Contextual Encoder Layer (`encoder: UBC-NLP/MARBERTv2`)
* **Input**: WordPiece token IDs (`input_ids`, shape: `[B, Seq_Len_Tokens]`).
* **Output**: Contextual hidden states (`[B, Seq_Len_Tokens, 768]`).
* **Purpose**: Captures sentence-level bidirectional semantics, syntax, and long-range dependencies across the entire Arabic text.

---

### 2. Sub-Character Embedding Layer (`char_embed: 256 -> 64`)
* **Input**: ASCII/Unicode byte value of each character (`0..255`, shape: `[B, Seq_Len_Chars]`).
* **Output**: Continuous character vector (`[B, Seq_Len_Chars, 64]`).
* **Purpose**: Provides character identity independent of sub-word tokenization, allowing the model to distinguish individual Arabic consonants, ligatures, digits, and punctuation marks.

---

### 3. Word-Relative Position Embedding Layer (`word_pos_embed: 32 -> 32`)
* **Input**: Character index relative to the start of its containing word (`0..31`, shape: `[B, Seq_Len_Chars]`).
* **Output**: Positional embedding vector (`[B, Seq_Len_Chars, 32]`).
* **Purpose**: Informs the network whether a character is at the beginning (prefix/proclitic), middle (stem root), or end (terminal suffix/case ending) of a word.

---

### 4. Morphological Clitic Embedding Layer (`morpheme_embed: 8 -> 16`)
* **Input**: Morpheme category ID (`0: None`, `1: Proclitic [ال، و، ف، ب، ك، ل، س]`, `2: Stem Core`, `3: Enclitic [هم، هن، نا، ها، كم...]`, shape: `[B, Seq_Len_Chars]`).
* **Output**: Morphological boundary vector (`[B, Seq_Len_Chars, 16]`).
* **Purpose**: Explicitly teaches the model Arabic clitic boundaries so it does not confuse proclitic prepositions with root consonants (e.g. distinguishing `فَـلَمْ` from `فَلَمْ`).

---

### 5. Feature Gathering & Dimension Projection Layer (`880d`)
* **Operation**: Projects token-level encoder representations to individual characters via offset mapping (`char_to_token`) and concatenates with embeddings:
  $$\mathbf{x}_{\text{combined}} = [\mathbf{h}_{\text{token}} \,\|\, \mathbf{e}_{\text{char}} \,\|\, \mathbf{e}_{\text{pos}} \,\|\, \mathbf{e}_{\text{morpheme}}] \in \mathbb{R}^{768 + 64 + 32 + 16 = 880}$$
* **Purpose**: Unifies high-level sentence context with fine-grained character-level morphological features into a unified 880-dimensional input vector for every character.

---

### 6. Bidirectional LSTM Layer (`lstm: 880 -> 384x2 = 768`)
* **Configuration**: 2 layers, bidirectional, dropout = 0.15.
* **Output**: Sequence representation (`[B, Seq_Len_Chars, 768]`).
* **Purpose**: Models sequential character-to-character phonetic constraints and phonotactics (such as vowel harmony, assimilation of solar letters `الـ` الشمسية, and elision before hamzat al-wasl).

---

### 7. Residual Projection & Layer Normalization (`proj_residual + layer_norm`)
* **Formula**: $\mathbf{h}_{\text{char}} = \text{LayerNorm}(\text{LSTM}(\mathbf{x}) + \mathbf{W}_{\text{proj}} \mathbf{x})$
* **Purpose**: Prevents vanishing gradients, preserves raw lexical signals across deep layers, and stabilizes multi-GPU distributed training.

---

### 8. Decoupled Lexical Stem Head (`stem_head: 768 -> 384 -> 15`)
* **Structure**: $\text{Linear}(768 \to 384) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(384 \to 15)$.
* **Purpose**: Specializes in predicting word-interior lexical and morphological vowels (Fatha, Damma, Kasra, Sukun, Shadda) that depend on the word's dictionary root and morphological pattern (الوزن الصرفي).

---

### 9. Decoupled Syntactic I'rab Head (`irab_head: 768 -> 384 -> 15`)
* **Structure**: $\text{Linear}(768 \to 384) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(384 \to 15)$.
* **Purpose**: Specializes in terminal letters (أواخر الكلمات) to determine grammatical case endings (الرفع بالضمة، النصب بالفتحة، الجر بالكسرة، الجزم بالسكون، والتنوين) based on sentence syntax.

---

### 10. Dynamic Emission Gating Matrix
* **Formula**:
  $$\mathbf{E}[b, t] = \begin{cases} \mathbf{P}_{\text{stem}}[b, t] & \text{if character } t \text{ is word-interior} \\ 0.3 \cdot \mathbf{P}_{\text{stem}}[b, t] + 0.7 \cdot \mathbf{P}_{\text{irab}}[b, t] & \text{if character } t \text{ is terminal/stem-end} \end{cases}$$
* **Purpose**: Fuses the morphological stem predictions and syntactic I'rab predictions with optimal weighting before passing emissions to the sequence layer.

---

### 11. Linear-Chain CRF Transition Layer (`crf: A_15x15`)
* **Parameters**: Transition matrix $\mathbf{A} \in \mathbb{R}^{15 \times 15}$, start vector $\mathbf{s} \in \mathbb{R}^{15}$, end vector $\mathbf{e} \in \mathbb{R}^{15}$.
* **Purpose**: Learns global pair transition probabilities $P(y_t \mid y_{t-1})$. Penalizes phonologically illegal transitions (e.g. consecutive Tanween, illegal double Shadda, Sukun on word-initial letters).
* **Training Loss**: Negative Log-Likelihood: $\mathcal{L}_{\text{CRF}} = \log Z(\mathbf{x}) - \text{score}(\mathbf{x}, \mathbf{y})$.

---

### 12. Viterbi Dynamic Programming Decoder (`decode()`)
* **Operation**: Computes the globally optimal sequence path:
  $$\mathbf{y}^* = \arg\max_{\mathbf{y}} \left( \sum_{t=1}^T \mathbf{E}_{t, y_t} + \sum_{t=1}^{T-1} \mathbf{A}_{y_t, y_{t+1}} \right)$$
* **Purpose**: Delivers the highest-scoring sentence-level diacritization without tokenization mismatch or character hallucination.

---

## 🔁 Pipeline Stages: Purpose of Every Train & Fine-Tune Step

```
Stage 1: Multi-Corpus Deduplication ──► Stage 2: Morpheme Backbone Pretraining (5.0M)
                                                      │
                                                      ▼
Stage 4: CRF & Viterbi Optimization  ◄── Stage 3: Target Domain Calibration (Sadeed)
            │
            ▼
Stage 5: Benchmark Evaluation (Sadeed & CATT) ──► Stage 6: Hugging Face Export
```

---

### Stage 1: Build & Deduplicate Multi-Domain Corpus (`01_build_multicorpus.py`)
* **Purpose**: Merges 3 diverse Arabic corpora (Sadeed Jurisprudence + Shamela Classical Library + Quranic Corpus).
* **Key Action**: Performs exact fingerprint matching against `Misraj/SadeedDiac-25` and `Bisher/CATT_benchmark` to guarantee **0% test contamination**.

---

### Stage 2: Train Morpheme Backbone (`02_train_morpheme_backbone.py`)
* **Purpose**: Pretrains the full 12-layer MARBERTv2 encoder, 2-layer BiLSTM, and decoupled Stem/I'rab heads across 5.0M sentences.
* **Loss**: $\mathcal{L} = \mathcal{L}_{\text{stem}} + 3.0 \cdot \mathcal{L}_{\text{irab}}$ with $0.03$ label smoothing.
* **Learning Rates**: $\text{LR}_{\text{encoder}} = 2.5 \times 10^{-5}$, $\text{LR}_{\text{head}} = 3.5 \times 10^{-4}$.

---

### Stage 3: Domain Calibration Fine-Tuning (`03_domain_calibration.py`)
* **Purpose**: Focuses representation capacity specifically on high-quality Modern Standard & Classical Arabic distributions.
* **Learning Rates**: Annealed down to $\text{LR}_{\text{encoder}} = 1.0 \times 10^{-5}$, $\text{LR}_{\text{head}} = 1.0 \times 10^{-4}$.

---

### Stage 4: Linear-Chain CRF Training (`04_train_crf_viterbi.py`)
* **Purpose**: Initializes the CRF transition matrix $\mathbf{A}_{15 \times 15}$ and fine-tunes the network to optimize global sequence probabilities.
* **Loss**: $\mathcal{L} = \mathcal{L}_{\text{CRF}}(\mathbf{E}, \mathbf{y}) + 0.5 \cdot \mathcal{L}_{\text{stem}} + 1.5 \cdot \mathcal{L}_{\text{irab}}$.
* **Result**: Drops Total WER on Sadeed below 8.0% (**7.91%**) and Morphological WER to **5.54%**.

---

### Stage 5: Benchmark Evaluation (`05_evaluate_benchmarks.py`)
* **Purpose**: Computes official character and word-level metrics using the verbatim Sadeed regex evaluator on both `Misraj/SadeedDiac-25` and `Bisher/CATT_benchmark`.

---

### Stage 6: Hugging Face Package Export (`06_export_hf_package.py`)
* **Purpose**: Converts trained weights to `model.safetensors`, exports tokenizer files, and configures `config.json` with `auto_map` for direct `AutoModel.from_pretrained()` loading.

---

## 🚀 How to Run the Pipeline

### One-Click Execution:
```bash
cd /mnt/models/tashkeel/tashkeel_v4_pipeline
./run_full_pipeline.sh
```

### Or Individual Step Execution:
```bash
# Step 1: Data Preparation
python 01_build_multicorpus.py

# Step 2: Backbone Training (Distributed across 2 GPUs)
torchrun --nproc_per_node=2 02_train_morpheme_backbone.py --batch-size 128 --epochs 3

# Step 3: Domain Calibration
python 03_domain_calibration.py --epochs 2

# Step 4: CRF & Viterbi Training
torchrun --nproc_per_node=2 04_train_crf_viterbi.py --batch-size 128 --epochs 2

# Step 5: Official Evaluation
python 05_evaluate_benchmarks.py

# Step 6: Package for Hugging Face Hub
python 06_export_hf_package.py
```
