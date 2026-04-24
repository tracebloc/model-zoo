# Text classification

Classify documents, reviews, tickets, or any short-to-medium-length text into predefined labels.

## Start here

**New to text classification?** Use [`pytorch/distilbert.py`](pytorch/distilbert.py). ~60% the size of BERT-base, ~97% of its accuracy — the strongest default when inference speed or model size matters.

If you want absolute maximum accuracy and can afford the compute, use [`pytorch/roberta_base.py`](pytorch/roberta_base.py).

## Models

| Model | Params | When to pick |
|---|---|---|
| [`distilbert.py`](pytorch/distilbert.py) | ~66M | Fast, pretrained; default for most tasks |
| [`bert_base_uncased.py`](pytorch/bert_base_uncased.py) | ~110M | Canonical BERT baseline; pretrained |
| [`roberta_base.py`](pytorch/roberta_base.py) | ~125M | Often beats BERT on downstream tasks |
| [`distilbert_scratch.py`](pytorch/distilbert_scratch.py) | ~66M | DistilBERT without pretrained weights; rarely the right choice |
| [`bert_base_uncased_scratch.py`](pytorch/bert_base_uncased_scratch.py) | ~110M | BERT from scratch; only for massive domain-specific corpora |
| [`simple_text.py`](pytorch/simple_text.py) | — | Embedding + dense layers; quick prototyping on small corpora |

## Dataset expectations

- **Input**: text strings; tokenization happens inside the model via HuggingFace tokenizers.
- **Labels**: integer class indices.
- **Batch size**: default 512.
