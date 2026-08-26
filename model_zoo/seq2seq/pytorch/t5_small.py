"""T5-small via HuggingFace, pretrained. Compact text-to-text transformer (~60M params) for seq2seq fine-tuning on a domain corpus.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained tensors are delivered
from the tracebloc model store as the training seed: upload the matched
``t5_small_weights.pkl`` sitting next to this file via ``weights=True``,
with the matching tokenizer as an explicitly named sibling (t5-small's
SentencePiece-Unigram tokenizer with ``</s>`` (eos) and ``<pad>`` already
defined)::

    user.upload_model("t5_small", weights=True,
                      tokenizer="t5_small_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: the sibling
seq2seq template uses a different vocabulary, and the SDK auto-attaches a
bare ``tokenizer.json`` to every model in the folder.)
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""
from transformers import AutoModelForSeq2SeqLM, AutoConfig

framework = "pytorch"
main_class = "MyModel"
category = "seq2seq"
model_type = ""
batch_size = 8
sequence_length = 128
license = "Apache-2.0"

# Architecture config for t5-small, inlined so the model builds with no
# config fetch. The SDK uploads the .py plus its named weight and tokenizer
# siblings — there is no config.json path — so the config lives here in the
# template.
CONFIG = {
    "model_type": "t5",
    "vocab_size": 32128,
    "d_model": 512,
    "d_kv": 64,
    "d_ff": 2048,
    "num_layers": 6,
    "num_decoder_layers": 6,
    "num_heads": 8,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "dropout_rate": 0.1,
    "layer_norm_epsilon": 1e-06,
    "initializer_factor": 1.0,
    "feed_forward_proj": "relu",
    "is_encoder_decoder": True,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "decoder_start_token_id": 0,
}


def MyModel():
    # Return the AutoModelForSeq2SeqLM directly (not wrapped in a custom
    # nn.Module) so the instance exposes ``.config`` and the SDK keeps it on
    # the HuggingFace path — shifting labels internally, building
    # decoder_input_ids itself, and resolving the tokenizer from the shipped
    # named sibling. The shared token embedding is tied across encoder,
    # decoder and lm_head (shared storage), as in upstream T5.
    config = AutoConfig.for_model(**CONFIG)
    return AutoModelForSeq2SeqLM.from_config(config)
