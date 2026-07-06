"""all-MiniLM-L6-v2 via HuggingFace, pretrained. Compact 6-layer BERT encoder (~22M params) for self-supervised contrastive sentence embeddings."""
from transformers import AutoModel

model_id = 'sentence-transformers/all-MiniLM-L6-v2'
# HF tokenizer distributed to every client as the federation's single
# source of truth (#805). Mandatory for HuggingFace NLP models; without it
# the SDK rejects the upload. Equal to model_id — all-MiniLM-L6-v2 ships its
# own WordPiece tokenizer with [PAD] and [CLS]/[SEP] already defined. Do NOT
# also ship a tokenizer.json for this model; the client loads it from the Hub.
tokenizer_id = 'sentence-transformers/all-MiniLM-L6-v2'
framework = "pytorch"
main_class = "MyModel"
category = "embeddings"
model_type = ""
batch_size = 16
sequence_length = 128
license = "Apache-2.0"


def MyModel():
    # Return the AutoModel (the bare encoder) directly, not wrapped in a custom
    # nn.Module, so the instance exposes ``.config`` and the SDK keeps it on the
    # HuggingFace path (tokenizer resolved from ``tokenizer_id``, no shipped
    # tokenizer.json expected). The encoder emits ``last_hidden_state`` shaped
    # ``(batch, seq_len, hidden)``; the training container mean-pools it
    # (attention-mask aware) into one embedding per example for the contrastive
    # objective. No classification / LM head — this is a plain text encoder.
    #
    # LoRA, if wanted, is selected in the training plan, never bundled here:
    # the package check rejects in-file PEFT.
    return AutoModel.from_pretrained(model_id)
