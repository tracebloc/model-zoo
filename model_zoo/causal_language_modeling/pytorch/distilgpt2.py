"""DistilGPT-2 via HuggingFace, pretrained. Compact GPT-2 (~82M params) for next-token fine-tuning on a domain corpus.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained tensors are delivered
from the tracebloc model store as the training seed: upload the matched
``distilgpt2_weights.pkl`` sitting next to this file via ``weights=True``,
with the matching tokenizer as an explicitly named sibling (distilgpt2's
GPT-2 byte-level BPE tokenizer; it defines ``<|endoftext|>`` as eos and the
client sets ``pad_token = eos_token``)::

    user.upload_model("distilgpt2", weights=True,
                      tokenizer="distilgpt2_tokenizer.json")

(A bare ``tokenizer.json`` cannot live in this directory: sibling
causal_language_modeling templates use a different vocabulary, and the SDK
auto-attaches a bare ``tokenizer.json`` to every model in the folder.)
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""
from transformers import AutoModelForCausalLM, AutoConfig

framework = "pytorch"
main_class = "MyModel"
category = "causal_language_modeling"
model_type = ""
batch_size = 8
# distilgpt2 supports up to 1024 positions; the causal-LM uploader caps
# sequence_length at max_position_embeddings=512.
sequence_length = 128
vocab_size = 50257
license = "Apache-2.0"

# Architecture config for distilgpt2, inlined so the model builds with no
# config fetch. The SDK uploads the .py plus its named weight and tokenizer
# siblings — there is no config.json path — so the config lives here in the
# template.
CONFIG = {
    "model_type": "gpt2",
    "vocab_size": vocab_size,
    "n_positions": 1024,
    "n_embd": 768,
    "n_layer": 6,
    "n_head": 12,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "layer_norm_epsilon": 1e-05,
    "initializer_range": 0.02,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "summary_activation": None,
    "summary_proj_to_labels": True,
    "summary_first_dropout": 0.1,
    "bos_token_id": 50256,
    "eos_token_id": 50256,
}


def MyModel():
    # Return the AutoModelForCausalLM directly (not wrapped in a custom
    # nn.Module) so the instance exposes ``.config`` and the SDK keeps it on
    # the HuggingFace path — shifting labels internally and resolving the
    # tokenizer from the shipped named sibling. The lm_head is tied to the
    # input embeddings (shared storage), as in upstream GPT-2.
    config = AutoConfig.for_model(**CONFIG)
    return AutoModelForCausalLM.from_config(config)
