"""DistilGPT-2 via HuggingFace, pretrained. Compact GPT-2 (~82M params) for next-token fine-tuning on a domain corpus."""
from transformers import AutoModelForCausalLM

model_id = 'distilgpt2'
# HF tokenizer distributed to every client as the federation's single
# source of truth (#805). Mandatory for HuggingFace NLP models; without it
# the SDK rejects the upload. Equal to model_id — distilgpt2 ships its own
# GPT-2 byte-level BPE tokenizer; the client sets pad_token = eos_token.
tokenizer_id = 'distilgpt2'
framework = "pytorch"
main_class = "MyModel"
category = "causal_language_modeling"
model_type = ""
batch_size = 8
# distilgpt2 supports up to 1024 positions; the causal-LM uploader caps
# sequence_length at max_position_embeddings=512.
sequence_length = 128
license = "Apache-2.0"


def MyModel():
    # Return the AutoModelForCausalLM directly (not wrapped in a custom
    # nn.Module) so the instance exposes ``.config`` and the SDK keeps it on
    # the HuggingFace path — shifting labels internally and resolving the
    # tokenizer from ``tokenizer_id`` rather than expecting a shipped
    # tokenizer.json.
    return AutoModelForCausalLM.from_pretrained(model_id)
