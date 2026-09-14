# Tokenizers shipped with the NLP templates

Training runs with the HuggingFace hub closed, so an NLP model trains with the
tokenizer that was **uploaded with it** and with nothing else. Every NLP template
in this zoo therefore names the file it ships in one module attribute:

```python
tokenizer_file = "simple_text_tokenizer.json"
```

The value is a bare filename, resolved in the template's own directory. Upload
the two together:

```python
user.upload_model("model_zoo/text_classification/pytorch/simple_text.py",
                  tokenizer="simple_text_tokenizer.json")
```

`tests/test_tokenizer_declaration.py` keeps the declaration honest: the file
exists, it is a HuggingFace `tokenizers` JSON, it carries the special tokens the
training engine requires for that task, its ids fit the template's embedding
table, and no tokenizer file in the directory is left unnamed.

## The two shapes a file takes

| kind | files | when |
|---|---|---|
| **the model's own tokenizer** | `<model>_tokenizer.json` beside an offline-migrated pretrained template (`bert_base_uncased_tokenizer.json`, `distilbert_tokenizer.json`, `qwen2_5_0_5b_tokenizer.json`, `t5_small_tokenizer.json`, …) | the architecture was pretrained with this exact vocabulary; the file is the published tokenizer of that checkpoint, with padding baked in |
| **the shared bert-vocab tokenizer** | `simple_*_tokenizer.json`, `masked_language_modeling/pytorch/tokenizer.json`, `embeddings_tokenizer.json`, `bert_vocab_tokenizer.json`, `seq2seq_tokenizer.json` | from-scratch templates that size their embedding table at `vocab_size = 30522` and need any consistent WordPiece vocabulary of that width |

## The shared bert-vocab files are verified copies, not generated files

They are the standard `bert-base-uncased` WordPiece tokenizer — the same 30522-entry
vocabulary as `text_classification/pytorch/bert_base_uncased_tokenizer.json`, entry
for entry, the same normalizer, pre-tokenizer, decoder and added tokens (`[PAD]`
id 0, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`) — saved with **padding baked in**
(`BatchLongest`, `pad_id=0`, `pad_token="[PAD]"`) so a training container needs
no configuration, in two variants:

* **encoder-style** copies keep the `[CLS] … [SEP]` `TemplateProcessing`
  post-processor (text / token / sentence-pair classification, masked language
  modeling, embeddings);
* **decoder-style** copies drop the post-processor, because causal language
  modeling and sequence-to-sequence models concatenate raw tokens and must not
  have sentence markers wrapped around every window.

Every copy is **the same tokenizer as `bert_base_uncased_tokenizer.json`**,
differing only in whether the post-processor is present — `tests/test_tokenizer_declaration.py`
asserts that field by field (normalizer, pre-tokenizer, decoder, the 30522-entry
model, added tokens, baked padding, truncation), and that the copies come in
exactly those two variants. Several copies are byte-for-byte duplicates under
different names, because a bare `tokenizer.json` is auto-detected by the SDK and
shipped for every model in its directory, so in a directory that mixes pretrained
templates with their own vocabularies the shared file carries a distinct name.

**Regeneration is not the recipe.** Re-saving a tokenizer with the `tokenizers`
library rewrites the file's schema and can renumber ids, and these files are
named by templates whose embedding tables are sized to them. If a copy ever has
to change, change the reference once, copy its bytes to the other names (dropping
the post-processor for the decoder-style ones), and let the test confirm every
copy still equals the reference.

## Special tokens the engine requires

The training engine validates a contributor file before training and refuses to
add a missing special token (adding one would mint an id past the embedding
table). Any one accepted surface form satisfies a role:

| role | accepted forms | required for |
|---|---|---|
| pad | `[PAD]`, `<pad>`, `<\|endoftext\|>`, `</s>` | every NLP task |
| mask | `[MASK]`, `<mask>` | masked language modeling |
