# Vendored contract: `object_detection_families.v1.json`

This is a **vendored, byte-identical copy** of the object-detection `model_type`
vocabulary the tracebloc training engine accepts. It is the single source of
truth for which `model_type` strings an OD model template may declare.

- **Upstream:** `tracebloc/tracebloc-engine` — `core/schema/object_detection_families.v1.json`
- **Published by:** tracebloc-engine#481 / PR #637 — the engine published this
  schema precisely so producers (this model-zoo, the backend's
  `MODEL_TYPE_CHOICES`) can assert in their own CI that what they emit the
  engine can route, instead of a user's experiment discovering the
  disagreement at run time (backend#1829).
- **Pinned ref:** `a6f2cd4dcbf415bfa27fa4029af295e04aa86682` (tracebloc-engine
  `develop`, 2026-08-27).

`tests/test_od_model_type_contract.py` reads this file — it needs no network.
It asserts every OD template's `model_type` is in `accepted_model_type_values`
(so the engine's `resolve_family()` can route it). The same set is, by the
backend's `global_meta/tests/test_od_families_contract.py`, exactly the OD
subset of `Experiment.MODEL_TYPE_CHOICES` — so membership also proves the value
is a valid backend choice.

## Refreshing this copy

tracebloc-engine is a private repo, so use an authenticated `gh` read (no
CI drift job is wired for it yet — that needs a cross-repo `contents:read`
token). From a checkout, with `gh` logged in:

```bash
gh api \
  "repos/tracebloc/tracebloc-engine/contents/core/schema/object_detection_families.v1.json?ref=<engine-sha>" \
  -H "Accept: application/vnd.github.raw" \
  > tests/contracts/tracebloc_engine/object_detection_families.v1.json
```

When adopting an upstream vocabulary change, bump the pinned ref above, refresh
this file, reconcile any OD template `model_type` declarations, and commit the
lot in one PR.
