# Vendored contract: `object_detection_families.v2.json`

This is a **vendored copy** of the object-detection `model_type` vocabulary the
tracebloc training engine accepts. It is the single source of truth for which
`model_type` strings an OD model template may declare.

It is identical to upstream in **every field the contract reads** —
`version`, `accepted_model_type_values`, each family's `family`, `aliases`,
`training_supported` and `handler`, and `not_accepted.examples` — and differs
from it in prose only: this repository is public, and the upstream prose cites
internal issues, so those citations are replaced with `(internal ref)` here.

- **Upstream:** `tracebloc-engine` — `core/schema/object_detection_families.v2.json`
- **Published by:** (internal ref) / (internal ref) — the engine published this
  schema precisely so producers (this model-zoo, the backend's
  `MODEL_TYPE_CHOICES`) can assert in their own CI that what they emit the
  engine can route, instead of a user's experiment discovering the
  disagreement at run time.
- **Pinned ref:** `generated_from` in the JSON: the engine commit, its branch,
  the blob sha of the schema there, and the digest of the contract fields.
- **Version:** v2. v1 published four accepted values; v2 publishes three —
  (internal ref) retired the `hf_transformer` (DETR) family and deleted v1
  upstream. Narrowing an accepted set is a **breaking** change for a consumer
  (a value it may still emit stops resolving), which is why it is a new version
  and not an edit in place, and why this repo's bump ships in the same PR as the
  deletion of the seven templates that declared it.

`tests/test_od_model_type_contract.py` reads this file — it needs no network.
It asserts every OD template's `model_type` is in `accepted_model_type_values`
(so the engine's `resolve_family()` can route it). The same set is, by the
backend's `global_meta/tests/test_od_families_contract.py`, exactly the OD
subset of `Experiment.MODEL_TYPE_CHOICES` — so membership also proves the value
is a valid backend choice.

## Refreshing this copy

`tools/contract_copies.py --write --ref develop` re-derives the contract fields
from the engine and re-pins `generated_from`; it keeps this copy's prose and
scrubs any upstream note it has to carry over.
`tests/test_vendored_contract_carries_no_internal_refs.py` still fails on a
citation, whatever wrote it. How the copy is held — offline on every pull
request, and weekly against the engine's branch of the same name — is in
`tools/contract_copies.py`.

When adopting an upstream vocabulary change, reconcile any OD template
`model_type` declarations in the same pull request.
