---
paths:
  - "tools/*.py"
  - "tests/mutation_selection.py"
  - "tests/*_rows.py"
  - "tests/test_*_guard.py"
  - "tests/test_folder_template_zips.py"
  - ".github/mutation-coverage.yml"
  - ".github/workflows/mutation-nightly.yml"
---
## Mutation rows run weekly, not on pull requests

The files listed under `rows:` in `.github/mutation-coverage.yml` hold the mutation rows
for the guards in `tools/`. `pytest tests/`, `make check` and CI's test jobs deselect the
rows; `make mutation-nightly` runs them and nothing else, and
`.github/workflows/mutation-nightly.yml` runs that target weekly and files an issue when
it fails. A pull request still runs each guard's behaviour tests, and
`mutation-coverage / check` still requires every guard to have a row, so a guard a pull
request breaks is caught by the next weekly run. `tests/mutation_selection.py` decides
which tests are rows: every test of a `*_rows.py` file, else the `test_row_*` and
`test_MUTATION_*` tests. Name a new row that way: a row named otherwise runs on every pull
request instead, and the weekly run refuses a rows file in which no test is a row.
