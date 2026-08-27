"""all-MiniLM-L6-v2 via HuggingFace, pretrained. Compact 6-layer BERT encoder (~22M params) for self-supervised contrastive sentence embeddings.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``minilm_weights.pkl`` sitting next to this file via ``weights=True``,
with the matching tokenizer as an explicitly named sibling::

    user.upload_model("minilm", weights=True,
                      tokenizer="minilm_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: the sibling
embeddings template uses a different tokenizer, and the SDK auto-attaches a
bare ``tokenizer.json`` to every model in the folder.)
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""

from transformers import AutoModel, AutoConfig

framework = "pytorch"
main_class = "MyModel"
category = "embeddings"
model_type = ""
batch_size = 16
sequence_length = 128
vocab_size = 30522
license = "Apache-2.0"

# Architecture config for sentence-transformers/all-MiniLM-L6-v2 (a 6-layer
# 384-hidden BERT encoder), inlined so the model builds with no config fetch.
# The SDK uploads the .py plus its named weight and tokenizer siblings —
# there is no config.json path — so the config lives here in the template.
CONFIG = {
    "model_type": "bert",
    "vocab_size": vocab_size,
    "hidden_size": 384,
    "num_hidden_layers": 6,
    "num_attention_heads": 12,
    "intermediate_size": 1536,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
}


def MyModel():
    # Return the AutoModel (the bare encoder) directly, not wrapped in a custom
    # nn.Module, so the instance exposes ``.config`` and the SDK keeps it on the
    # HuggingFace path. The encoder emits ``last_hidden_state`` shaped
    # ``(batch, seq_len, hidden)``; the training container mean-pools it
    # (attention-mask aware) into one embedding per example for the contrastive
    # objective. No classification / LM head — this is a plain text encoder.
    #
    # LoRA, if wanted, is selected in the training plan, never bundled here:
    # the package check rejects in-file PEFT.
    config = AutoConfig.for_model(**CONFIG)
    return AutoModel.from_config(config)
