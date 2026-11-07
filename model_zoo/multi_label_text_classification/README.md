# Multi-label text classification

Assign every class that applies to a text or sequence: a protein's functions (GO terms), a document's topics, a ticket's affected components. Each class is its own yes/no decision, so one sample can carry several classes at once, and the classes do not compete for one slot as they do in text classification.

## Start here

**Classifying protein sequences?** Use [`pytorch/esm2_35m.py`](pytorch/esm2_35m.py), ESM-2 pretrained on UniRef50 protein sequences, with its tokenizer [`pytorch/esm2_35m_tokenizer.json`](pytorch/esm2_35m_tokenizer.json):

```python
user.upload_model("model_zoo/multi_label_text_classification/pytorch/esm2_35m.py",
                  weights=True, tokenizer="esm2_35m_tokenizer.json")
```

## Models

| Model | Params | When to pick |
|---|---|---|
| [`esm2_35m.py`](pytorch/esm2_35m.py) | ~34M | Protein sequences; pretrained, small enough to fine-tune fully |

## Dataset expectations

- **Input**: texts, or sequences (one protein per sample, residues as letters); tokenization happens inside the training container with the tokenizer the template names in `tokenizer_file` (uploaded with the model — see [`../TOKENIZERS.md`](../TOKENIZERS.md)).
- **Labels**: every class that applies, joined by `;` in one cell (e.g. `GO:0005737;GO:0005634`); at least one class per sample.
- **Output**: `(B, output_classes)` raw logits, one per class — no sigmoid in the model. `output_classes` is the dataset's number of classes.
- **Sequence length**: 512 tokens (`esm2_35m`), so it trains within the smallest client's memory; longer sequences are cut. ESM-2 was pretrained on up to 1024; raise the sequence length in the training plan on a client with more memory.
- **Batch size**: 2 (`esm2_35m`).
