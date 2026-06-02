"""Wide but shallow HF BERT for MLM, ~35M params. PEFT auto-detects LoRA targets — no fallback needed."""
from transformers import BertForMaskedLM, BertConfig

framework = "pytorch"
main_class = "WideMiniMLM"
category = "masked_language_modeling"
model_type = ""
batch_size = 32
sequence_length = 128
vocab_size = 30522


class WideMiniMLM(BertForMaskedLM):
    """4-layer, 512-dim, 8-head HuggingFace BERT for masked language modeling.

    ~35M parameters. Subclasses BertForMaskedLM so PEFT can auto-detect
    LoRA target modules (query, key, value) without fallback scanning.
    """

    def __init__(self, vocab_size=vocab_size):
        config = BertConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            vocab_size=vocab_size,
            max_position_embeddings=512,
        )
        super().__init__(config)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs.logits
