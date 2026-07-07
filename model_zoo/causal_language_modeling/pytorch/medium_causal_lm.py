"""Mid-size 6-layer GPT-style decoder for causal LM, ~35M params. Balanced speed/quality for moderate datasets."""
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "MediumCausalLM"
category = "causal_language_modeling"
model_type = ""
batch_size = 32
sequence_length = 256
vocab_size = 30522
license = "Apache-2.0"


class MediumCausalLM(nn.Module):
    """6-layer, 512-dim decoder-only transformer for causal language modeling.

    ~35M parameters (the tied output projection keeps the count down). A middle
    ground between the lightweight SimpleCausalLM and the GPT-2-scale
    NanoGPTScratch — suitable for datasets with 100K–1M sequences. Uses a causal
    attention mask (next-token prediction) and ties the output projection to the
    token embedding (GPT-style weight tying).

    The model emits raw logits shaped ``(batch, seq_len, vocab_size)``. It does
    NOT shift the labels: the training container applies the single next-token
    shift (logits at position t are scored against the token at position t+1).
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        dropout=0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.max_position_embeddings = max_position_embeddings
        self.embed_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
            # The nested-tensor fast path can bypass the attention mask, which
            # would break causality — keep it disabled.
            enable_nested_tensor=False,
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.lm_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        # Weight tying: LM decoder shares weights with token embeddings (GPT-style).
        self.lm_decoder.weight = self.word_embeddings.weight
        self.lm_bias = nn.Parameter(torch.zeros(vocab_size))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
                if module.out_proj.weight is not None:
                    module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
                if module.out_proj.bias is not None:
                    module.out_proj.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        input_ids = input_ids.long()
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=self.max_position_embeddings - 1)

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.embed_dropout(x)

        # Causal mask: position i may not attend to any position j > i.
        # Bool mask (True = ignore) to match the bool padding mask below.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = self.decoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        x = self.final_layer_norm(x)
        logits = self.lm_decoder(x) + self.lm_bias
        return logits
