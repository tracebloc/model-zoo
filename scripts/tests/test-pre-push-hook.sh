#!/bin/sh
# test-pre-push-hook.sh — behaviour tests for this repo's own Makefile targets,
# driven by running real `make` against a copy of the Makefile in a throwaway git
# repo. Pure POSIX sh, no bats/pytest dependency, so `make test-hooks` runs it
# anywhere.
#
# Covers: guard-toolchain fails on a tool it cannot run and passes when it can.
# The pre-push installer and the hook it writes are the generated
# .github/install-hooks.mk, byte for byte the org's canonical copy, and are tested
# where that copy lives.
set -eu

# Hermetic: ignore the developer's global/system git config; every git here sees
# only local config.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null

REPO_ROOT=$(unset CDPATH; cd -- "$(dirname -- "$0")/../.." && pwd)
MAKEFILE="$REPO_ROOT/Makefile"
fails=0
check() { if [ "$1" = "$2" ]; then echo "  ok: $3"; else echo "  FAIL: $3 (got '$1', want '$2')"; fails=$((fails + 1)); fi; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
git init -q .
git config user.email t@t; git config user.name t
cp "$MAKEFILE" ./Makefile
# The Makefile includes the generated installer, so the copy needs it beside it.
mkdir -p .github && cp "$REPO_ROOT/.github/install-hooks.mk" .github/
# 9) guard-toolchain itself: fails when a tool cannot run, passes when all can.
# Driven with a stub PYTHON so it needs no real venv and stays fast.
gtbin="$(mktemp -d)"
printf '#!/bin/sh\nexit 0\n' > "$gtbin/py-ok"; chmod +x "$gtbin/py-ok"
printf '#!/bin/sh\nexit 1\n' > "$gtbin/py-bad"; chmod +x "$gtbin/py-bad"
if make -s guard-toolchain PYTHON="$gtbin/py-ok" >/dev/null 2>&1; then rc=0; else rc=$?; fi
check "$rc" "0" "guard-toolchain passes when the toolchain runs"
if make -s guard-toolchain PYTHON="$gtbin/py-bad" >/dev/null 2>&1; then rc=0; else rc=$?; fi
check "$( [ "$rc" != 0 ] && echo nonzero || echo zero )" "nonzero" "guard-toolchain fails when a tool cannot run"
rm -rf "$gtbin"

echo "--- Makefile behaviour tests: $( [ "$fails" -eq 0 ] && echo ALL GREEN || echo "$fails FAILED" ) ---"
[ "$fails" -eq 0 ]
