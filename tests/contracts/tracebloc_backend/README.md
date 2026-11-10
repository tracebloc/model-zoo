# Vendored contract: `model_type_choices.v1.json`

The complete `model_type` vocabulary the platform accepts, across **all** task
types — derived from `Experiment.MODEL_TYPE_CHOICES`, the `choices=` of the
`model_type` field on `Experiment`.

- **Upstream:** `backend` — `experiments/models/experiment.py`
- **Pinned ref:** `generated_from` in the JSON: the producer commit, its branch,
  the blob sha of each source file there, and the digest of the derived fields.
- **Why it exists:** `model_type` is a Django `ChoiceField`. A value outside the
  choice set is refused at experiment creation with a **400, before the model
  file is looked at** — and nothing in that error names the template as the
  cause. Three keypoint templates shipped `model_type = "transformer"`, which
  was never a member, so each was unusable as shipped with no hint why
  (internal ref).

`tests/test_zoo_model_type_contract.py` reads this file — it needs no network
and no Django. It asserts every template's declared `model_type` is storable.

## This is a SUPERSET, not a routing vocabulary

Membership here means **storable**, not well-routed for a given category. Two
narrower vocabularies sit underneath it and are checked separately:

| scope | vocabulary | checked by |
|---|---|---|
| all task types (storable) | this file, 16 values | `tests/test_zoo_model_type_contract.py` |
| object detection (routable) | `../tracebloc_engine/object_detection_families.v2.json`, 3 values | `tests/test_od_model_type_contract.py` |
| keypoint detection (routable) | `{rcnn, heatmap}` special-cased, everything else → direct regression | the engine's `_infer_model_type`; no vendored schema published |

So an OD template must satisfy **both** this file and the engine schema, and the
OD schema's accepted set is a subset of this one. A test asserts that
containment, so a backend narrowing that stranded an OD value would be caught
here rather than at run time.

## `""` is accepted, and that is not the same as "safe"

The field is `blank=True` / `null=True` and
`ExperimentSerializer.validate_model_type` coerces a falsy value to `default`,
so an empty declaration is stored rather than refused — which is why most zoo
templates declare `""`. That is a statement about the **backend** only. The
engine's keypoint `_infer_model_type` falls an empty value through to a final
`RCNN_FAMILY` fallback, so `""` will mis-route a keypoint model that is not an
R-CNN. Declare the family explicitly there.

## Refreshing this copy

Both copies in this directory are written by `tools/contract_copies.py`; how
they are held, and how to refresh them, is in that module's docstring and in
[the drift section below](#how-these-copies-are-held). Never hand-edit a
contract field: the offline check fails on a field that disagrees with
`generated_from.fields_sha256`.

Narrowing the set is a **breaking** change for this repo (a value a template
still declares stops being storable): reconcile any template declaration the
refresh strands in the same pull request.

---

# Vendored contract: `upload_import_allowlist.v1.json`

The top-level packages an uploaded model file may import: `ALLOWED_PYTHON_PACKAGE_IMPORTS`
in the backend's upload-scan Bandit plugin (TBT001).

- **Upstream:** `backend` — `bandit_tracebloc_model_validation/bandit_plugins/import_names.py`
- **Pinned ref:** `generated_from` in the JSON, as above.
- **Why it exists:** the scan refuses an upload whose file imports any other top-level name,
  **standard library included**. `from subprocess import run` or `import os` is enough. Nothing in
  this repo checked it, so templates importing names outside the list shipped and could not be
  uploaded.

`tests/test_upload_import_allowlist.py` reads this file. It is stdlib only and runs in every CI
framework job.

Some of these modules are admitted member by member upstream: importing one does not
admit every name in it, and this copy lists only the top-level names.

## How these copies are held

- **On every pull request, offline**: `tests/test_contract_copies_held.py` (in the required
  `category-contract` check) holds each copy to the producer commit, source blobs and field digest
  its `generated_from` records.
- **Weekly, against the producer**: `.github/workflows/contract-drift.yml` re-derives the copies on
  this repo's `develop`, `staging` and `main` from the producer's branch of the same name. So the
  copy that reaches `main` must equal the producer's `main`, which is what prod runs.
  It is required nowhere and opens one issue when a leg disagrees.

Refresh from the producer branch, then reconcile the templates in the same pull request:

```bash
python3 tools/contract_copies.py --ref develop                 # what differs
python3 tools/contract_copies.py --write --ref develop --only tracebloc_backend/upload_import_allowlist.v1.json
```
