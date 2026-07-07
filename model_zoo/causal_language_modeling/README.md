# Causal language modeling

Train a decoder-only (GPT-style) language model that predicts the next token from the tokens before it. Use it to pretrain a small LM on a domain corpus, or to fine-tune one for instruction/SFT-style completion.

## Start here

**Just validating the pipeline?** Use [`pytorch/simple_causal_lm.py`](pytorch/simple_causal_lm.py) — a 4-layer decoder that trains in seconds and confirms data, tokenizer, and shift logic are wired up correctly.

For real from-scratch pretraining on a domain corpus, use [`pytorch/nanogpt_style_scratch.py`](pytorch/nanogpt_style_scratch.py) (GPT-2-small shape). To **fine-tune a pretrained** model instead of training from scratch, use [`pytorch/distilgpt2.py`](pytorch/distilgpt2.py).

## Models

| Model | Params | When to pick |
|---|---|---|
| [`simple_causal_lm.py`](pytorch/simple_causal_lm.py) | ~19M | Smoke test; quick iteration and pipeline validation |
| [`medium_causal_lm.py`](pytorch/medium_causal_lm.py) | ~35M | Moderate datasets (100K–1M sequences); weight-tied |
| [`nanogpt_style_scratch.py`](pytorch/nanogpt_style_scratch.py) | ~109M | From-scratch pretraining on domain corpora; GPT-2-small shape |
| [`distilgpt2.py`](pytorch/distilgpt2.py) | ~82M | Fine-tune a pretrained GPT-2 (HuggingFace) instead of training from scratch |

The first three are plain `nn.Module` decoders (no pretrained weights, no in-file PEFT) that output raw logits shaped `(batch, seq_len, vocab_size)`. `distilgpt2.py` is a HuggingFace `AutoModelForCausalLM` returned directly — it ships pretrained weights and shifts labels internally. LoRA, if wanted, is selected in the training plan — never bundled into the model file.

## Dataset expectations

- **Input**: `.txt` files — plain text for pretraining, or `prompt\tcompletion` lines for SFT.
- **Labels**: position-aligned and **unshifted**. The model does **not** shift labels; the training container applies the single next-token shift (logits at position *t* are scored against the token at position *t+1*).
- **Batch size**: 32 (simple/medium), 16 (nanogpt).

## Tokenizer

The tokenizer is the federation's single source of truth, distributed to every client (issue #805). How it is supplied depends on whether the model is HuggingFace or custom:

- **`distilgpt2.py` (HuggingFace)** declares a `tokenizer_id`; the client loads the matching GPT-2 tokenizer from the Hub. Upload it with no tokenizer argument:

  ```python
  user.upload_model("model_zoo/causal_language_modeling/pytorch/distilgpt2.py")
  ```

- **The three custom (non-HuggingFace) models** use [`pytorch/bert_vocab_tokenizer.json`](pytorch/bert_vocab_tokenizer.json) — a WordPiece tokenizer over the standard 30522-token bert-base-uncased vocabulary with the encoder-style `[CLS]…[SEP]` auto-wrapping removed (a causal LM concatenates raw tokens). `[SEP]` (id 102) serves as the end-of-text/eos token and `[PAD]` (id 0) as pad; for decoder-only training the client sets `pad_token = eos_token`.

  This file is deliberately **not** named `tokenizer.json`: the SDK auto-detects a sibling `tokenizer.json` and ships it for *every* model in the directory, which would override `distilgpt2.py`'s `tokenizer_id` and silently feed it a BERT tokenizer. Because of the non-default name it is not auto-detected, so pass it explicitly for each custom model:

  ```python
  user.upload_model(
      "model_zoo/causal_language_modeling/pytorch/simple_causal_lm.py",
      tokenizer="model_zoo/causal_language_modeling/pytorch/bert_vocab_tokenizer.json",
  )
  ```

  (Same for `medium_causal_lm.py` and `nanogpt_style_scratch.py`.) The vocabulary fits every custom model's embedding table.
