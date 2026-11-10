# Vendored contract: `sklearn_model_types.v1.json`

The `model_type` values the tracebloc SDK's **sklearn tabular upload path**
accepts, per family. Unlike its siblings this file is **derived, not
byte-copied** — the SDK publishes no schema, so the vocabulary is extracted
from its source by AST.

- **Upstream:** `tracebloc-py-package` —
  `tracebloc/upload/skl_tabular_{base,classifier,regression}.py`
- **Pinned ref:** `generated_from` in the JSON: the SDK commit, its branch,
  each source file's blob sha there, and the digest of `accepted_by_family`.
  The hand-written mirror in
  `tests/test_zoo_sklearn_model_type_trainable.py::_as_sdk_sees_it` is
  re-checked against the source whenever a refresh moves a blob.

## Why this is a DIFFERENT gate from `tracebloc_backend/model_type_choices.v1.json`

That one says what the backend can **store**; this one says what survives the
SDK's `model_func_checks`. **A value can be storable and still be
untrainable** — an empty declaration is stored as `'default'`, but the SDK
coerces `""` to `None` (`tracebloc/validation/rewriter.py::_parse_constant_rhs`)
and no accepted set contains `None`, so `average_estimators` raises
*"model type None is not supported for Sklearn"*. That storable-but-untrainable
gap is exactly how the EBM declaration incident shipped, and it is why this
file exists as a second contract rather than an extension of the first.

Note `'default'` is **also** absent from every set below, so a blank was never
going to work by that route either.

## Refreshing this copy

The derivation — the keys of each `_ensemble_prefixes` dict, unioned with every
string literal compared against `model_type` inside `average_estimators` and
`_average_family_specific` — is `derive_sklearn_types` in
`tools/contract_copies.py`, which also holds the copy offline on every pull
request and weekly against the SDK's branch of the same name:

```bash
python3 tools/contract_copies.py --ref develop --show
python3 tools/contract_copies.py --write --ref develop --only tracebloc_sdk/sklearn_model_types.v1.json
```

If the sets change, re-check the mirror in
`test_zoo_sklearn_model_type_trainable.py`, reconcile any sklearn template
`model_type` declarations, and commit the lot in one pull request.

`tests/test_zoo_sklearn_model_type_trainable.py` reads this file and needs no
network.
