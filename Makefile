# Makefile for tracebloc/model-zoo — uniform entry points (backend#1606).
#
# Every active tracebloc repo exposes the SAME three targets, so "run
# your tests before you push" stops being a rule you can only obey with
# per-repo tribal knowledge:
#
#   make check      lint + fast tests.   Budget: under 60 s.
#   make check-all  everything CI runs (bar the CI-only heavy suites).
#   make setup      install what those targets need.
#
# This file is a THIN WRAPPER over ci.yml. It introduces no new tool,
# no new config and no new rule. When ci.yml changes, change the
# matching line here.
#
# Uses whatever python/pytest is on PATH, i.e. your active virtualenv.

.DEFAULT_GOAL := help

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

# ci.yml fans the same `pytest tests/` out across four framework
# environments. Locally you install one; FRAMEWORK picks which
# .github/requirements/<name>.txt `make setup` installs.
FRAMEWORK ?= pytorch

.PHONY: help
help:
	@echo "tracebloc/model-zoo — make targets"
	@echo
	@echo "  check       ruff + the model-contract tests — run this before every push"
	@echo "  check-all   the same, verbose — CI's extra width is four framework envs"
	@echo "  setup       pip install the lint + FRAMEWORK requirement sets"
	@echo
	@echo "  individual: lint test"
	@echo
	@echo "  CI runs 'pytest tests/' four times, once per framework env:"
	@echo "  pytorch, tensorflow, sklearn, survival. Locally you have one."
	@echo "  Pick it with:  make setup FRAMEWORK=tensorflow"

# ---- check: the pre-push tier ------------------------------------
#
# The whole suite here is two files of contract tests, so the fast tier
# and the full tier differ only in verbosity. What CI has that a laptop
# does not is BREADTH — the same tests against four framework installs
# — and that is a fan-out no Makefile should try to reproduce in one
# environment.
.PHONY: check
check: lint test
	@echo "==> check: green (CI additionally runs these under 4 framework envs)"

.PHONY: check-all
check-all: lint
	$(PYTEST) tests/ -v
	@echo "==> check-all: green (CI additionally runs these under 4 framework envs)"

# setup: dependencies only. No pre-commit / pre-push hook is installed
# here — that is a later step of backend#1606.
.PHONY: setup
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r .github/requirements/lint.txt
	$(PYTHON) -m pip install -r .github/requirements/$(FRAMEWORK).txt
	@echo "==> setup: lint + $(FRAMEWORK) requirements installed; run 'make check'"

# ---- individual targets ------------------------------------------

# lint: ci.yml's `ruff` job, same selection and same path. Deliberately
# narrower than the org default: this tree is example model code, and
# the three families here are the ones that mean "this file is broken".
.PHONY: lint
lint:
	ruff check --select=F401,F821,E9 model_zoo/

# test: ci.yml's `pytest tests/` (one framework env at a time locally).
.PHONY: test
test:
	$(PYTEST) tests/ -q
