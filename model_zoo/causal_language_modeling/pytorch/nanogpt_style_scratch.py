"""GPT-2-small-scale decoder for causal LM, ~109M params. Train from scratch on domain corpora (next-token pretraining)."""
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "NanoGPTScratch"
category = "causal_language_modeling"
model_type = ""
batch_size = 16
sequence_length = 256
vocab_size = 30522
license = "Apache-2.0"


class NanoGPTScratch(nn.Module):
    """GPT-2-small-scale decoder-only transformer for causal language modeling.

    ~109M parameters with default settings (12 layers, 768 hidden, 12 heads;
    GPT-2-small shape, with a 30522-token vocab and tied output projection).
    Pre-norm transformer blocks with a causal attention mask and GPT-style
    weight tying. Designed for from-scratch pretraining on domain corpora where
    off-the-shelf language models lack coverage.

    The model emits raw logits shaped ``(batch, seq_len, vocab_size)``. It does
    NOT shift the labels: the training container applies the single next-token
    shift (logits at position t are scored against the token at position t+1).

    Small enough to train fully and average across federated clients (no
    pretrained weights, no parameter-efficient adapters baked in — any such
    choice belongs in the training plan, not the model file).
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024,
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
            norm_first=True,  # pre-norm, like GPT-2
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
                # Q/K/V projections stored as raw Parameters, not nn.Linear.
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
