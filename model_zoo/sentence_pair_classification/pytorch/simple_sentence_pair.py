"""Minimal token-embedding + mean-pool + dense sentence-pair classifier.

Pick for quick prototyping on small labeled pair corpora. The client tokenizes
each example as ``tokenizer(text_a, text_b)``, so a single ``input_ids`` sequence
arrives already holding both sentences joined by ``[SEP]`` (and marked by
``token_type_ids``). This bag-of-embeddings model consumes that concatenated
sequence directly and emits ``(batch, num_classes)`` logits — no label shifting,
no PEFT wrappers (LoRA is selected in the training plan, not baked into the
model file)."""
import torch.nn as nn
import torch.nn.functional as F

framework = "pytorch"
main_class = "SimpleSentencePairClassifier"
category = "sentence_pair_classification"
model_type = ""
batch_size = 512
# mandatory for non huggingface model
sequence_length = 64
output_classes = 2
license = "Apache-2.0"

# Must cover the training tokenizer's vocabulary. The shipped WordPiece
# tokenizer.json (and the client's default bert-base-uncased) use
# vocab_size=30522 — token IDs beyond the embedding table cause
# index-out-of-bounds errors at training time.
_VOCAB_SIZE = 30522


class SimpleSentencePairClassifier(nn.Module):
    """Bag-of-embeddings pair classifier with a small two-layer head.

    NLP inputs are integer token IDs (Long tensors) emitted by the tokenizer,
    not float embeddings — so the model needs a real ``nn.Embedding`` front
    end. For a sentence pair the tokenizer already interleaves both sentences
    into one ``input_ids`` sequence (``text_a [SEP] text_b``), so the same
    front end works unchanged: the IDs index the embedding table, the per-token
    vectors are mean-pooled (attention-mask aware) into a single vector, and the
    dense head projects to the class logits. Output is ``(batch, num_classes)``.
    """

    def __init__(self, vocab_size=_VOCAB_SIZE, embed_dim=128, num_classes=output_classes):
        super().__init__()
        # Token-embedding front end: maps Long token IDs -> dense vectors,
        # matching the sibling non-HF NLP models (simple_text / simple_token).
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)  # First dense layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Tokenizers emit integer token IDs. The upload smoke-test casts inputs
        # to the model's float dtype for loss-less models, so input_ids may
        # arrive as a Float tensor; nn.Embedding requires Long indices.
        input_ids = input_ids.long()

        # (batch, seq_len, embed_dim)
        embedded = self.word_embeddings(input_ids)

        # Mean-pool over the sequence into one vector per example, ignoring
        # padding positions when an attention mask is provided.
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(embedded.dtype)
            summed = (embedded * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        else:
            pooled = embedded.mean(dim=1)

        # Apply a fully connected layer and ReLU activation
        x = F.relu(self.fc1(pooled))
        # Output layer
        x = self.fc2(x)
        return x
