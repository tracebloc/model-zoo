# Vendored contract: `sklearn_model_types.v1.json`

The `model_type` values the tracebloc SDK's **sklearn tabular upload path**
accepts, per family. Unlike its siblings this file is **derived, not
byte-copied** — the SDK publishes no schema, so the vocabulary is extracted
from its source by AST.

- **Upstream:** `tracebloc-py-package` —
  `tracebloc/upload/skl_tabular_{base,classifier,regression}.py`
- **Pinned ref:** see `generated_from.ref` in the JSON (with `ref_branch` and
  `ref_date`). That is also the ref the hand-written mirror in
  `tests/test_zoo_sklearn_model_type_trainable.py::_as_sdk_sees_it` was
  verified against.

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

`tracebloc-py-package` is private, so use an authenticated `gh` read. This
derives the sets mechanically — **never hand-transcribe them**, which is how the
earlier OD vocabulary drift shipped. From a checkout, with `gh` logged in:

```bash
SDK_REF=<sdk-sha>            # record this in generated_from.ref
mkdir -p /tmp/sdkderive && for f in base classifier regression; do
  gh api "repos/tracebloc-py-package/contents/tracebloc/upload/skl_tabular_${f}.py?ref=${SDK_REF}" \
    -H "Accept: application/vnd.github.raw" > "/tmp/sdkderive/skl_tabular_${f}.py"
done

python3 - <<'PY'
import ast, glob, json, collections
acc = collections.defaultdict(set)
FAM = {"classifier": "classifier", "regression": "regression"}
for path in sorted(glob.glob("/tmp/sdkderive/skl_tabular_*.py")):
    fam = next((v for k, v in FAM.items() if path.endswith(f"{k}.py")), None)
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        # 1. keys of the _ensemble_prefixes dict (the ensemble-packed types)
        if isinstance(node, ast.Assign) and any(
            getattr(t, "attr", getattr(t, "id", None)) == "_ensemble_prefixes"
            for t in node.targets
        ) and isinstance(node.value, ast.Dict):
            for k in node.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    for f in ([fam] if fam else FAM.values()):
                        acc[f].add(k.value)
        # 2. literals compared against `model_type` in the averaging strategies
        if isinstance(node, ast.FunctionDef) and node.name in (
            "average_estimators", "_average_family_specific"
        ):
            for cmp in (n for n in ast.walk(node) if isinstance(n, ast.Compare)):
                if getattr(cmp.left, "id", None) == "model_type":
                    for c in cmp.comparators:
                        vals = c.elts if isinstance(c, (ast.Tuple, ast.List, ast.Set)) else [c]
                        for v in vals:
                            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                                for f in ([fam] if fam else FAM.values()):
                                    acc[f].add(v.value)
print(json.dumps({k: sorted(v) for k, v in acc.items()}, indent=2))
PY
```

Diff that output against `accepted_by_family`. If it differs: bump
`generated_from.ref`/`ref_date`, update the sets, **re-check the mirror** in
`test_zoo_sklearn_model_type_trainable.py` against the new ref, reconcile any
sklearn template `model_type` declarations, and commit the lot in one PR.

`tests/test_zoo_sklearn_model_type_trainable.py` reads this file and needs no
network.
