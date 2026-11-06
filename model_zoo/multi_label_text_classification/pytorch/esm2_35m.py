"""ESM-2 35M (Meta AI, 2022). A protein language model pretrained on UniRef50, fine-tuned here to tag each sequence with every class it carries (GO terms, enzyme classes, localisations). Each class gets its own logit; the platform trains it with a sigmoid per class (``bcewithlogits``), so any number of classes can be on for one sequence. Small enough to fine-tune fully; select LoRA-only fine-tuning in the training plan if you want federated averaging to sync only the adapter tensors.

Offline variant: the architecture is built from the inlined config below —
no hub model id, no config fetch, no download at build time, so the template
constructs anywhere, network or not. The pretrained encoder tensors are
delivered from the tracebloc model store as the training seed: upload the
matched ``esm2_35m_weights.pkl`` sitting next to this file via
``weights=True``, with the matching tokenizer as an explicitly named sibling::

    user.upload_model("esm2_35m", weights=True,
                      tokenizer="esm2_35m_tokenizer.json")

The tokenizer is ESM-2's own vocabulary as a ``tokenizer.json``: one token per
residue, wrapped in ``<cls>`` ... ``<eos>``, padded with ``<pad>``.
See ``tools/prep_offline_weights.py`` for producing and verifying the
matched weight file.
"""

from transformers import AutoModelForSequenceClassification, AutoConfig

# The task head is NOT carried by the hosted seed: the seed holds the backbone,
# and the head initialises fresh from output_classes, which is where the
# dataset's class count lands. Derived by tools/derive_seed_excluded.py.
SEED_EXCLUDED_PREFIXES = ("classifier.out_proj.",)

framework = "pytorch"
main_method = "MyModel"
license = "MIT"
category = "multi_label_text_classification"
model_type = ""
tokenizer_file = "esm2_35m_tokenizer.json"
batch_size = 8
# ESM-2 was pretrained on 1,022 residues plus <cls> and <eos>.
sequence_length = 1024
output_classes = 10

# Architecture config for facebook/esm2_t12_35M_UR50D, inlined so the model
# builds with no config fetch. The SDK uploads the .py plus its named weight
# and tokenizer siblings — there is no config.json path — so the config lives
# here in the template.
CONFIG = {
    "model_type": "esm",
    "vocab_size": 33,
    "hidden_size": 480,
    "num_hidden_layers": 12,
    "num_attention_heads": 20,
    "intermediate_size": 1920,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "max_position_embeddings": 1026,
    "position_embedding_type": "rotary",
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-05,
    "emb_layer_norm_before": False,
    "token_dropout": True,
    "pad_token_id": 1,
    "mask_token_id": 32,
    "classifier_dropout": None,
    "is_folding_model": False,
    "problem_type": "multi_label_classification",
}


def MyModel(num_classes=output_classes):
    config = AutoConfig.for_model(**CONFIG, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
