import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel

try:
    from .configuration_tashkeel import TashkeelConfig
    from .linear_chain_crf import LinearChainCRF
except ImportError:
    from configuration_tashkeel import TashkeelConfig
    from linear_chain_crf import LinearChainCRF

DIACRITICS_LIST = [
    '',        # 0: None
    'َ',       # 1: Fatha
    'ُ',       # 2: Damma
    'ِ',       # 3: Kasra
    'ْ',       # 4: Sukun
    'ً',       # 5: Tanween Fath
    'ٌ',       # 6: Tanween Damm
    'ٍ',       # 7: Tanween Kasr
    'ّ',       # 8: Shadda alone
    'َّ',      # 9: Shadda + Fatha
    'ُّ',      # 10: Shadda + Damma
    'ِّ',      # 11: Shadda + Kasra
    'ًّ',      # 12: Shadda + Tanween Fath
    'ٌّ',      # 13: Shadda + Tanween Damm
    'ٍّ',      # 14: Shadda + Tanween Kasr
]

DIAC_TO_ID = {d: i for i, d in enumerate(DIACRITICS_LIST)}
NUM_CLASSES = len(DIACRITICS_LIST)

MORPHEME_PROCLITIC = 1
MORPHEME_STEM = 2
MORPHEME_ENCLITIC = 3

PROCLITICS_LIST = ["وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل", "س"]
ENCLITICS_LIST = ["هما", "كما", "هم", "هن", "كم", "كن", "نا", "ها", "وا", "ون", "ين", "ان", "ات", "ه", "ك", "ي"]
STANDARD_ARABIC_LETTERS = set("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي")

def is_arabic_letter(c: str) -> bool:
    return c in STANDARD_ARABIC_LETTERS

def segment_word_morphemes(word: str):
    w_len = len(word)
    if w_len <= 3 or not any(is_arabic_letter(c) for c in word):
        return [(0, w_len, word, MORPHEME_STEM)]
        
    p_len = 0
    for p in PROCLITICS_LIST:
        if word.startswith(p) and len(word) - len(p) >= 2:
            p_len = len(p)
            break
            
    e_len = 0
    remaining = word[p_len:]
    for e in ENCLITICS_LIST:
        if remaining.endswith(e) and len(remaining) - len(e) >= 2:
            e_len = len(e)
            break
            
    stem_len = w_len - p_len - e_len
    spans = []
    if p_len > 0:
        spans.append((0, p_len, word[:p_len], MORPHEME_PROCLITIC))
    spans.append((p_len, p_len + stem_len, word[p_len:p_len+stem_len], MORPHEME_STEM))
    if e_len > 0:
        spans.append((p_len + stem_len, w_len, word[p_len+stem_len:], MORPHEME_ENCLITIC))
        
    return spans


class TashkeelV3ForDiacritization(PreTrainedModel):
    config_class = TashkeelConfig
    _tied_weights_keys = []
    _keys_to_ignore_on_load_missing = []
    _keys_to_ignore_on_load_unexpected = []

    def __init__(self, config=None, model_name=None, num_classes=15):
        if config is None:
            config = TashkeelConfig(base_model_name_or_path=model_name or "UBC-NLP/MARBERTv2", num_classes=num_classes)
        super().__init__(config)

        base_model = getattr(config, "base_model_name_or_path", "UBC-NLP/MARBERTv2")
        num_classes = getattr(config, "num_classes", 15)

        encoder_config = AutoConfig.from_pretrained(base_model)
        self.encoder = AutoModel.from_config(encoder_config)
        hidden_size = self.encoder.config.hidden_size

        self.char_embed = nn.Embedding(256, 64)
        self.word_pos_embed = nn.Embedding(32, 32)
        self.morpheme_embed = nn.Embedding(8, 16)

        input_feat_dim = hidden_size + 64 + 32 + 16  # 880

        self.lstm = nn.LSTM(
            input_size=input_feat_dim,
            hidden_size=384,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )

        self.layer_norm = nn.LayerNorm(768)
        self.proj_residual = nn.Linear(input_feat_dim, 768)

        self.stem_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_classes),
        )

        self.irab_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_classes),
        )

        self.crf = LinearChainCRF(num_tags=num_classes)
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        char_to_token,
        char_mask,
        char_ids=None,
        word_pos_ids=None,
        morpheme_type_ids=None,
        is_irab_pos=None,
    ):
        token_hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        max_tok_idx = token_hidden.size(1) - 1
        char_to_token = torch.clamp(char_to_token, 0, max(max_tok_idx, 0))

        if char_ids is not None:
            char_ids = torch.clamp(char_ids, 0, 255)
        if word_pos_ids is not None:
            word_pos_ids = torch.clamp(word_pos_ids, 0, 31)
        if morpheme_type_ids is not None:
            morpheme_type_ids = torch.clamp(morpheme_type_ids, 0, 7)

        expanded_indices = char_to_token.unsqueeze(-1).expand(-1, -1, token_hidden.size(-1))
        char_token_feats = torch.gather(token_hidden, 1, expanded_indices)

        c_emb = self.char_embed(char_ids)
        p_emb = self.word_pos_embed(word_pos_ids)
        m_emb = self.morpheme_embed(morpheme_type_ids)

        char_combined = torch.cat([char_token_feats, c_emb, p_emb, m_emb], dim=-1)

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(char_combined)
        h_char = self.layer_norm(lstm_out + self.proj_residual(char_combined))

        stem_logits = self.stem_head(h_char)
        irab_logits = self.irab_head(h_char)

        emissions = stem_logits.clone()
        if is_irab_pos is not None:
            irab_mask = is_irab_pos.unsqueeze(-1)
            fused = 0.3 * stem_logits + 0.7 * irab_logits
            emissions = torch.where(irab_mask, fused, stem_logits)

        return {
            "stem_logits": stem_logits,
            "irab_logits": irab_logits,
            "emissions": emissions,
        }

    @torch.inference_mode()
    def diacritize(self, text_or_list, tokenizer=None, device=None, batch_size=64, use_viterbi=True):
        is_single = isinstance(text_or_list, str)
        texts = [text_or_list] if is_single else list(text_or_list)

        if device is None:
            device = next(self.parameters()).device
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")

        self.eval()
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            offsets_batch = enc["offset_mapping"]
            if hasattr(offsets_batch, "tolist"):
                offsets_batch = offsets_batch.tolist()
            num_tokens = enc["input_ids"].shape[1]

            B = len(batch_texts)
            max_char_len = max(len(t) for t in batch_texts)

            char_to_token = torch.zeros((B, max_char_len), dtype=torch.long, device=device)
            char_mask = torch.zeros((B, max_char_len), dtype=torch.bool, device=device)
            char_ids = torch.zeros((B, max_char_len), dtype=torch.long, device=device)
            word_pos_ids = torch.zeros((B, max_char_len), dtype=torch.long, device=device)
            morpheme_type_ids = torch.zeros((B, max_char_len), dtype=torch.long, device=device)
            is_irab_pos = torch.zeros((B, max_char_len), dtype=torch.bool, device=device)

            for b in range(B):
                text = batch_texts[b]
                offsets = offsets_batch[b]
                tok_idx = 0
                words = text.split(" ")
                char_in_word_map = []
                morpheme_map = []
                is_stem_end_map = []

                for w in words:
                    wl = len(w)
                    spans = segment_word_morphemes(w)
                    word_morpheme_types = [0] * wl
                    word_stem_ends = [False] * wl
                    for start, end, chunk, m_type in spans:
                        for idx_c in range(start, end):
                            word_morpheme_types[idx_c] = m_type
                        if m_type == MORPHEME_STEM and end > start:
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
                    char_ids[b, c_idx] = min(ord(text[c_idx]), 255)
                    if c_idx < len(char_in_word_map):
                        word_pos_ids[b, c_idx] = char_in_word_map[c_idx]
                        morpheme_type_ids[b, c_idx] = morpheme_map[c_idx]
                        is_irab_pos[b, c_idx] = is_stem_end_map[c_idx]

            outputs = self(
                input_ids,
                attention_mask,
                char_to_token,
                char_mask,
                char_ids=char_ids,
                word_pos_ids=word_pos_ids,
                morpheme_type_ids=morpheme_type_ids,
                is_irab_pos=is_irab_pos,
            )

            emissions = outputs["emissions"]

            if use_viterbi:
                paths = self.crf.decode(emissions, char_mask)
            else:
                paths = torch.argmax(emissions, dim=-1).cpu().tolist()

            for b in range(B):
                text = batch_texts[b]
                p_row = paths[b]
                out_chars = []
                for c_i, c in enumerate(text):
                    out_chars.append(c)
                    if is_arabic_letter(c) and c_i < len(p_row):
                        label_id = p_row[c_i]
                        diac = DIACRITICS_LIST[label_id]
                        if diac:
                            out_chars.append(diac)
                results.append("".join(out_chars))

        return results[0] if is_single else results
