# Vendored contract: `model_type_choices.v1.json`

The complete `model_type` vocabulary the platform accepts, across **all** task
types — derived from `Experiment.MODEL_TYPE_CHOICES`, the `choices=` of the
`model_type` field on `metaApi.models.Experiment`.

- **Upstream:** `backend` — `metaApi/models/Experiment.py`
- **Pinned ref:** `daa607e5c7e1d051eb4ee7b8ffcd023f3b5c0371` (`develop`, 2026-09-07)
- **Why it exists:** `model_type` is a Django `ChoiceField`. A value outside the
  choice set is refused at experiment creation with a **400, before the model
  file is looked at** — and nothing in that error names the template as the
  cause. Three keypoint templates shipped `model_type = "transformer"`, which
  was never a member, so each was unusable as shipped with no hint why
  (model-zoo#273).

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

`backend` is a private repo, so use an authenticated `gh` read. The file is
**generated, not copied** — the values are resolved from the AST so a
hand-transcription cannot drift (a hand-copied list is how the OD vocabulary
drift shipped):

```bash
sha=$(gh api repos/backend/commits/develop --jq .sha)
gh api "repos/backend/contents/metaApi/models/Experiment.py?ref=$sha" \
  -H "Accept: application/vnd.github.raw" > /tmp/Experiment.py
python3 - "$sha" <<'PY'
import ast, json, sys
src = open("/tmp/Experiment.py").read()
cls = next(n for n in ast.walk(ast.parse(src))
           if isinstance(n, ast.ClassDef) and n.name == "Experiment")
consts = {t.id: n.value.value
          for n in cls.body if isinstance(n, ast.Assign)
          for t in n.targets
          if isinstance(t, ast.Name) and t.id.startswith("MODEL_TYPE_")
          and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
choices = next(n.value for n in cls.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "MODEL_TYPE_CHOICES"
                       for t in n.targets))
print(sys.argv[1])
print(sorted(consts[e.elts[0].id] for e in choices.elts))
PY
```

Then update `accepted_model_type_values`, `choices`, and `generated_from.ref` in
the JSON, reconcile any template declaration the change strands, and commit the
lot in one PR. Narrowing the set is a **breaking** change for this repo (a value
a template still declares stops being storable), which is what the `version`
field is for.
