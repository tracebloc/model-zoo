# Sentence-pair classification

Classify a *pair* of sentences into predefined labels — natural language
inference (entailment / neutral / contradiction), paraphrase / duplicate-question
detection, answer selection, and semantic-similarity buckets.

## Start here

**New to sentence-pair classification?** Use [`pytorch/bert_base_uncased.py`](pytorch/bert_base_uncased.py).
BERT is the strongest default for pair tasks because it was pretrained with
segment embeddings (`token_type_ids`) that let the model tell the two sentences
apart.

## Models

| Model | Params | When to pick |
|---|---|---|
| [`bert_base_uncased.py`](pytorch/bert_base_uncased.py) | ~110M | Canonical pair-classification baseline; pretrained, uses segment embeddings |
| [`simple_sentence_pair.py`](pytorch/simple_sentence_pair.py) | — | Embedding + dense layers; quick prototyping on small pair corpora |

## Dataset expectations

- **Input**: two text strings per example (`text_a`, `text_b`). The dataset
  stores them tab-separated; the client tokenizes them together as
  `tokenizer(text_a, text_b)`, joining them with `[SEP]` and marking each with
  `token_type_ids`. Tokenization happens inside the model's tokenizer.
- **Labels**: integer class indices (e.g. 0/1 for paraphrase, 0/1/2 for NLI).
- **Batch size**: default 512.

## Tokenizer

HuggingFace models (`bert_base_uncased.py`) load their tokenizer from the Hub
via the declared `tokenizer_id`. The non-HF `simple_sentence_pair.py` ships
[`pytorch/simple_sentence_pair_tokenizer.json`](pytorch/simple_sentence_pair_tokenizer.json)
(WordPiece, 30522-token BERT vocab, with `[PAD]`/`[CLS]`/`[SEP]`/`[UNK]`) and is
uploaded explicitly:

```python
user.upload_model("simple_sentence_pair", tokenizer="simple_sentence_pair_tokenizer.json")
```
