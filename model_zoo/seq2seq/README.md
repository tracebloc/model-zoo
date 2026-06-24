# Sequence-to-sequence (seq2seq)

Train an encoder-decoder model that maps a source sequence to a target sequence: translation, summarization, paraphrasing, or any text-to-text task. A bidirectional encoder reads the source; a causal decoder generates the target while cross-attending to the encoder output.

## Start here

**Just validating the pipeline?** Use [`pytorch/simple_seq2seq.py`](pytorch/simple_seq2seq.py) — a small from-scratch encoder-decoder transformer that confirms data, tokenizer, and the decoder-input/label shift are wired up correctly.

To **fine-tune a pretrained** text-to-text model instead of training from scratch, use [`pytorch/t5_small.py`](pytorch/t5_small.py).

## Models

| Model | Params | When to pick |
|---|---|---|
| [`simple_seq2seq.py`](pytorch/simple_seq2seq.py) | ~43M | From-scratch baseline; pipeline validation and quick iteration |
| [`t5_small.py`](pytorch/t5_small.py) | ~60M | Fine-tune a pretrained T5 (HuggingFace) instead of training from scratch |

`simple_seq2seq.py` is a plain `nn.Module` (no pretrained weights, no in-file PEFT) that outputs raw logits shaped `(batch, target_seq_len, vocab_size)`. `t5_small.py` is a HuggingFace `AutoModelForSeq2SeqLM` returned directly — it ships pretrained weights and builds `decoder_input_ids` / shifts labels internally. LoRA, if wanted, is selected in the training plan — never bundled into the model file.

## Dataset expectations

- **Input**: tab-separated `source\ttarget` text lines (one example per line).
- **`decoder_input_ids`**: the target sequence shifted right by one (a start token prepended). The training container builds these and passes them to `forward(..., decoder_input_ids=...)`.
- **Labels**: position-aligned and **unshifted**. The model does **not** shift labels; the container scores the logits at position *t* against the target token at position *t*.
- **Batch size**: 16 (`simple_seq2seq`), 8 (`t5_small`).

## Tokenizer

The tokenizer is the federation's single source of truth, distributed to every client (issue #805). How it is supplied depends on whether the model is HuggingFace or custom:

- **`t5_small.py` (HuggingFace)** declares a `tokenizer_id`; the client loads the matching T5 SentencePiece tokenizer from the Hub (it already defines `</s>` as eos and `<pad>` as pad). Upload it with no tokenizer argument:

  ```python
  user.upload_model("model_zoo/seq2seq/pytorch/t5_small.py")
  ```

- **`simple_seq2seq.py` (custom)** uses [`pytorch/seq2seq_tokenizer.json`](pytorch/seq2seq_tokenizer.json) — a WordPiece tokenizer over the standard 30522-token bert-base-uncased vocabulary with the encoder-style `[CLS]…[SEP]` auto-wrapping removed. `[PAD]` (id 0) is the pad token, `[CLS]` (id 101) serves as the decoder start token, and `[SEP]` (id 102) as the end-of-sequence token. Its max token id (30521) fits the model's embedding table.

  This file is deliberately **not** named `tokenizer.json`: the SDK auto-detects a sibling `tokenizer.json` and ships it for *every* model in the directory, which would override `t5_small.py`'s `tokenizer_id` and silently feed it a BERT tokenizer. Because of the non-default name it is not auto-detected, so pass it explicitly:

  ```python
  user.upload_model(
      "model_zoo/seq2seq/pytorch/simple_seq2seq.py",
      tokenizer="model_zoo/seq2seq/pytorch/seq2seq_tokenizer.json",
  )
  ```
