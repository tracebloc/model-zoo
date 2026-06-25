"""Minimal encoder-decoder transformer for seq2seq, ~43M params. From-scratch model for sequence-to-sequence training and pipeline validation."""
import torch
import torch.nn as nn

framework = "pytorch"
main_class = "SimpleSeq2Seq"
category = "seq2seq"
model_type = ""
batch_size = 16
sequence_length = 128
vocab_size = 30522
license = "Apache-2.0"


class SimpleSeq2Seq(nn.Module):
    """4+4-layer encoder-decoder transformer for sequence-to-sequence tasks.

    ~43M parameters with default settings (the tied embedding / output
    projection keeps the count down). A standard Transformer: a bidirectional
    encoder reads the source sequence and a causal decoder generates the target
    sequence, attending to the encoder output via cross-attention. Suitable for
    translation, summarization, and other text-to-text tasks. Intended as a
    light from-scratch baseline, not production quality.

    Source and target share one token embedding (the data contract uses a
    single tokenizer for both sides) and the output projection is tied to that
    embedding (T5/Transformer-base style weight tying).

    The model emits raw logits shaped ``(batch, target_seq_len, vocab_size)``.
    It does NOT shift the labels: the training container builds
    ``decoder_input_ids`` (the target shifted right by one) and scores the
    logits at position t against the target token at position t.
    """

    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=512,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        dropout=0.1,
    ):
        super().__init__()
        self.shared_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.encoder_position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.decoder_position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.max_position_embeddings = max_position_embeddings
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            # The nested-tensor fast path can silently drop the padding mask;
            # keep it disabled so padded source tokens stay masked.
            enable_nested_tensor=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Weight tying: output projection shares the shared token embedding.
        self.lm_head.weight = self.shared_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)

    def _embed(self, input_ids, position_embeddings):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=self.max_position_embeddings - 1)
        x = self.shared_embeddings(input_ids) + position_embeddings(position_ids)
        return self.embed_dropout(x)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, **kwargs):
        input_ids = input_ids.long()
        # If the container does not supply decoder_input_ids (e.g. a quick
        # smoke run), fall back to the source so the forward pass still works.
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        decoder_input_ids = decoder_input_ids.long()

        # Encoder: bidirectional, only the padding mask applies.
        src = self._embed(input_ids, self.encoder_position_embeddings)
        if attention_mask is not None:
            # TransformerEncoder expects True = ignore for the padding mask.
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Decoder: causal self-attention + cross-attention over the encoder memory.
        tgt = self._embed(decoder_input_ids, self.decoder_position_embeddings)
        tgt_len = decoder_input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=decoder_input_ids.device),
            diagonal=1,
        )
        x = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits
