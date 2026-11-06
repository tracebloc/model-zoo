# Text regression

Predict a number from a text or sequence: a protein's stability, fluorescence or binding affinity, a review's score, a document's reading level. Each sample has one number to predict, in its own units.

## Start here

**Predicting a property of protein sequences?** Use [`pytorch/esm2_35m.py`](pytorch/esm2_35m.py), ESM-2 pretrained on UniRef50 protein sequences, with its tokenizer [`pytorch/esm2_35m_tokenizer.json`](pytorch/esm2_35m_tokenizer.json):

```python
user.upload_model("model_zoo/text_regression/pytorch/esm2_35m.py",
                  weights=True, tokenizer="esm2_35m_tokenizer.json")
```

## Models

| Model | Params | When to pick |
|---|---|---|
| [`esm2_35m.py`](pytorch/esm2_35m.py) | ~34M | Protein sequences; pretrained, small enough to fine-tune fully |

## Dataset expectations

- **Input**: texts, or sequences (one protein per sample, residues as letters); tokenization happens inside the training container with the tokenizer the template names in `tokenizer_file` (uploaded with the model — see [`../TOKENIZERS.md`](../TOKENIZERS.md)).
- **Labels**: one number per text or sequence.
- **Output**: `(B, 1)`, the predicted number, with no activation in the model; `output_classes` is 1. The platform trains it with `mse` (the default) or `l1`, and reports scores in the target's own units.
- **Sequence length**: 1024 tokens (`esm2_35m`), ESM-2's pretraining length; longer sequences are cut.
- **Batch size**: 8 (`esm2_35m`).
