"""T5-small via HuggingFace, pretrained. Compact text-to-text transformer (~60M params) for seq2seq fine-tuning on a domain corpus."""
from transformers import AutoModelForSeq2SeqLM

model_id = 't5-small'
# HF tokenizer distributed to every client as the federation's single
# source of truth (#805). Mandatory for HuggingFace NLP models; without it
# the SDK rejects the upload. Equal to model_id — t5-small ships its own
# SentencePiece tokenizer with </s> (eos) and <pad> already defined.
tokenizer_id = 't5-small'
framework = "pytorch"
main_class = "MyModel"
category = "seq2seq"
model_type = ""
batch_size = 8
sequence_length = 128
license = "Apache-2.0"


def MyModel():
    # Return the AutoModelForSeq2SeqLM directly (not wrapped in a custom
    # nn.Module) so the instance exposes ``.config`` and the SDK keeps it on
    # the HuggingFace path — shifting labels internally, building
    # decoder_input_ids itself, and resolving the tokenizer from
    # ``tokenizer_id`` rather than expecting a shipped tokenizer.json.
    return AutoModelForSeq2SeqLM.from_pretrained(model_id)
