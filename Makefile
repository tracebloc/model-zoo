# Makefile for tracebloc/model-zoo — uniform entry points.
#
# Every active tracebloc repo exposes the SAME three targets, so "run
# your tests before you push" stops being a rule you can only obey with
# per-repo tribal knowledge:
#
#   make check      lint + fast tests.   Budget: under 60 s.
#   make check-all  everything CI runs (bar the CI-only heavy suites).
#   make setup      install what those targets need, and a git pre-push hook
#                   that runs `make check` (skip once with --no-verify).
#
# This file is a THIN WRAPPER over ci.yml. It introduces no new tool,
# no new config and no new rule. When ci.yml changes, change the
# matching line here.
#
# Uses whatever python/pytest is on PATH, i.e. your active virtualenv.

.DEFAULT_GOAL := help

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

# Lint tools are invoked through $(PYTHON) -m, not as bare commands on
# PATH. `setup` pip-installs them into $(PYTHON)'s environment at a
# pinned version; a bare `ruff` would resolve to whatever happens to be
# first on PATH — a homebrew build, another venv — and the pin the
# comment promises would silently not be the thing that ran. (Bugbot,
# (internal ref).)

# ci.yml fans the same `pytest tests/` out across three framework
# environments. Locally you install one; FRAMEWORK picks which
# .github/requirements/<name>.txt `make setup` installs.
FRAMEWORK ?= pytorch

.PHONY: help
help:
	@echo "tracebloc/model-zoo — make targets"
	@echo
	@echo "  check       ruff + the model-contract tests — run this before every push"
	@echo "  check-all   the same, verbose — CI's extra width is four framework envs"
	@echo "  setup       pip install the lint + FRAMEWORK requirement sets; installs the pre-push hook"
	@echo "  install-hooks  (re)install the git pre-push hook that runs 'make check'"
	@echo
	@echo "  individual: lint test index"
	@echo
	@echo "  CI runs 'pytest tests/' three times, once per framework env:"
	@echo "  pytorch, sklearn, survival. Locally you have one."
	@echo "  Pick it with:  make setup FRAMEWORK=sklearn"

# ---- check: the pre-push tier ------------------------------------
#
# The whole suite here is two files of contract tests, so the fast tier
# and the full tier differ only in verbosity. What CI has that a laptop
# does not is BREADTH — the same tests against four framework installs
# — and that is a fan-out no Makefile should try to reproduce in one
# environment.
# guard-toolchain: can this shell actually run `make check`? A GUI/IDE git client
# (VS Code, Tower, GitKraken) launches the pre-push hook on a thin PATH that has
# system `make` and often a bare `python3`, but NOT the virtualenv these checks
# were installed into — so `make check` hard-fails on "No module named ruff" with
# no --no-verify to escape (the make-vs-toolchain shape of
# (internal ref)). `check` depends on this and the hook runs it first, so a push
# from such a client degrades to a skip instead of a hard block.
#
# It asks the TOOLS, not the DEPENDENCIES: whether $(PYTHON) can run the linters
# and test runner check invokes, not whether every package is installed. A
# genuinely broken venv still fails `make check` when run from a real shell.
.PHONY: guard-toolchain
guard-toolchain:
	@command -v "$(PYTHON)" >/dev/null 2>&1 || { \
	  echo "not on PATH: $(PYTHON) — activate your virtualenv, then run: make setup"; \
	  exit 1; }
	@missing=''; \
	for m in ruff pytest; do \
	  "$(PYTHON)" -m "$$m" --version >/dev/null 2>&1 || missing="$$missing $$m"; \
	done; \
	[ -z "$$missing" ] || { \
	  echo "$(PYTHON) cannot run:$$missing — activate your virtualenv, then run: make setup"; \
	  exit 1; }

.PHONY: check
check: guard-toolchain lint test
	@echo "==> check: green (CI additionally runs these under 3 framework envs)"

.PHONY: check-all
check-all: lint
	$(PYTEST) tests/ -v
	@echo "==> check-all: green (CI additionally runs these under 3 framework envs)"

# setup: install dependencies, then install the git pre-push hook — the
# later step of (internal ref) — via the install-hooks target below.
.PHONY: setup
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r .github/requirements/lint.txt
	$(PYTHON) -m pip install -r .github/requirements/test.txt -r .github/requirements/$(FRAMEWORK).txt
	@echo "==> setup: lint + $(FRAMEWORK) requirements installed; run 'make check'"
	@$(MAKE) --no-print-directory install-hooks

# test-hooks: run the Makefile's own behaviour tests (guard-toolchain) on demand;
# the installer and its hook are the generated snippet's, tested where it lives.
# Not a `check` dependency: it runs real `make` in throwaway repos, so an
# environment quirk (old git, noexec /tmp) could block a local push on something
# CI never sees. Run it directly with `make test-hooks`.
.PHONY: test-hooks
test-hooks:
	@sh scripts/tests/test-pre-push-hook.sh

# The one pre-push installer, generated: `make install-hooks` writes the hook that runs
# `make check`, `make hooks-status` says whether it is current. Never edit
# .github/install-hooks.mk here; it is rendered from the org's canonical copy.
include .github/install-hooks.mk

# ---- individual targets ------------------------------------------

# lint: ci.yml's `ruff` job, same selection and same path. Deliberately
# narrower than the org default: this tree is example model code, and
# the three families here are the ones that mean "this file is broken".
.PHONY: lint
lint:
	$(PYTHON) -m ruff check --select=F401,F821,E9 model_zoo/

# test: ci.yml's `pytest tests/` (one framework env at a time locally).
.PHONY: test
test:
	$(PYTEST) tests/ -q

# index: rebuild model_zoo/index.v1.json after changing a template's
# batch_size, architecture or estimator. tests/test_zoo_index.py fails until the
# rebuilt file is committed. Rows whose framework is not installed here are
# carried from the committed file; CI recomputes them.
.PHONY: index
index:
	$(PYTHON) tools/build_index.py
