# Vendored contract: `object_detection_families.v2.json`

This is a **vendored copy** of the object-detection `model_type` vocabulary the
tracebloc training engine accepts. It is the single source of truth for which
`model_type` strings an OD model template may declare.

It is identical to upstream in **every field the contract reads** —
`accepted_model_type_values`, `families`, `not_accepted`, the version keys — and
differs from it in prose only: this repository is public, and the upstream
`description` and `supersedes` strings cite internal issues, so those citations
are replaced with `(internal ref)` here. An earlier version of this paragraph
claimed the copy was *byte-identical*; it has not been since the repository went
public, and the refresh recipe below still told you to overwrite the file with
raw upstream bytes. Both are corrected.

- **Upstream:** `tracebloc-engine` — `core/schema/object_detection_families.v2.json`
- **Published by:** (internal ref) / (internal ref) — the engine published this
  schema precisely so producers (this model-zoo, the backend's
  `MODEL_TYPE_CHOICES`) can assert in their own CI that what they emit the
  engine can route, instead of a user's experiment discovering the
  disagreement at run time.
- **Pinned ref:** `060f339e71c4cff7e9fb595b8c5b445e03b503ea` — the **merge**
  commit on the engine's `develop`, 2026-09-01. It previously named
  `320fe41f`, the commit on the (since deleted) branch: a branch SHA and a
  merge SHA are indistinguishable by eye, and only the second stays reachable.
  `320fe41f` is not an ancestor of the engine's `develop`, so the refresh
  recipe below would eventually have 404'd on a documented procedure with no
  way to tell whether the pin or the procedure was at fault. The vendored
  contract fields are the same at both commits — nothing about the vocabulary
  changed, only the pin's reachability.
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

The engine lives in a private repo, so use an authenticated `gh` read (no CI
drift job is wired for it yet — that needs a cross-repo `contents:read` token).
The path must be owner-qualified; `repos/tracebloc-engine/...` — what this
recipe said before — is not a repository and 404s. From a checkout, with `gh`
logged in:

```bash
gh api \
  "repos/tracebloc/tracebloc-engine/contents/core/schema/object_detection_families.v2.json?ref=<engine-sha>" \
  -H "Accept: application/vnd.github.raw" \
  > tests/contracts/tracebloc_engine/object_detection_families.v2.json
```

**Then scrub the prose before you commit.** This repository is public and the
upstream strings cite internal issues; replace each citation with
`(internal ref)`, leaving every other field byte-for-byte as upstream wrote it.
`tests/test_vendored_contract_carries_no_internal_refs.py` fails if you skip
this, so a forgotten scrub is caught here rather than published.

When adopting an upstream vocabulary change, bump the pinned ref above, refresh
this file, scrub it, reconcile any OD template `model_type` declarations, and
commit the lot in one PR.
