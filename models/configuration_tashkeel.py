from transformers import PretrainedConfig

class TashkeelConfig(PretrainedConfig):
    model_type = "tashkeel_v3"

    def __init__(
        self,
        base_model_name_or_path="UBC-NLP/MARBERTv2",
        num_classes=15,
        char_embed_dim=64,
        word_pos_embed_dim=32,
        morpheme_embed_dim=16,
        lstm_hidden_size=384,
        lstm_num_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.num_classes = num_classes
        self.char_embed_dim = char_embed_dim
        self.word_pos_embed_dim = word_pos_embed_dim
        self.morpheme_embed_dim = morpheme_embed_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
