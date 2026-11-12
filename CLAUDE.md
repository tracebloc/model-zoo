<!-- claude-budget: armed -->
<!-- root = commands, the area map, rules every session needs; module rules in `<dir>/CLAUDE.md` or `.claude/rules/` with `paths:`; reference in `docs/`; history in commit messages -->
# CLAUDE.md

Public zoo of model files for tracebloc training: `model_zoo/` holds the models by task and
framework, `tools/` the scripts and guards that build and check them, `tests/` their tests.

## Before you push

`make check` runs the checks CI runs, and the pre-push hook runs it before every push.
`make setup` installs the hook in this clone, which covers every worktree of it, and
`make hooks-status` says whether the installed hook is current.

## Map

- `model_zoo/`: the model files, by task and framework.
- `tools/`: build and publish scripts and the guards; mutation rows: `.claude/rules/mutation-rows.md`.
- `tests/`: `pytest tests/`; mutation rows run weekly, see `.claude/rules/mutation-rows.md`.
- `.github/workflows/`: CI; `.github/mutation-coverage.yml` declares guards and rows.
