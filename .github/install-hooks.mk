# .github/install-hooks.mk -- the one pre-push hook installer. GENERATED: never edit it in
# a repo. The org's onboarding script renders this file byte for byte into every train
# repo (`onboard_repo.py --render-install-hooks`), and the pipeline audit compares it.
#
# A repo's Makefile includes it and its `setup` runs `install-hooks`:
#
#     include .github/install-hooks.mk
#     setup: ...
#     	@$(MAKE) --no-print-directory install-hooks
#
#   make install-hooks   write the pre-push hook into this clone's hooks directory
#   make hooks-status    exit 0 only when the installed hook is this file's hook
#   make check-stamped   run `make check`, and stamp the tree it passed on, so the push of
#                        that tree skips the hook's run (call it instead of a bare make check)
#   make pass-record-toolchain   print the toolchain a pass record carries (THE TOOLCHAIN)
#
# THE STAMP. A pass of `make check` is recorded as HEAD's tree hash, an empty file in
# `git rev-parse --git-path tracebloc-check-stamps` (this checkout's git directory, never
# the work tree), and only when make check exited 0 and `git status` showed nothing
# tracked or untracked (not ignored) differing from HEAD, before and after the run, on
# the same HEAD tree. The newest 20 stamps are kept. The hook skips `make check` only
# when the tree of every commit it is pushing has a stamp; a tree it cannot read runs it.
# The hook's own passing run stamps too, so a retried push is free.
#
# THE PASS RECORD. The stamp of a clean pass also holds the pass record of its tree, one
# line: `pass-record/1 tree=<tree> toolchain=<make pass-record-toolchain>`. A stamp whose
# toolchain cannot be told holds no record and still skips the hook's run. When the hook
# lets a push to github.com through, it publishes the record of every pushed commit whose
# tree has one, as a commit status on that commit (context `tracebloc/pass-record`, the
# record as its description). The remote has no commit until the push ends, so a detached
# step waits for this push's git to exit, at most 120 seconds, then makes one `gh api`
# call per commit. No `gh` on PATH, a `gh` with no login, or a push that never lands
# leaves no record, and CI then runs every check. A pull request's required job skips
# what a record covers only when the record is for the head's own tree and the job's own
# toolchain and was published by a person's account; the gates never read one.
#
# THE TOOLCHAIN. Each runtime the checkout pins in a version file (`.python-version`;
# `.nvmrc` or `.node-version`; go.mod's `go` line), as this machine resolves it, cut to the
# pin's precision: `python=3.12`. Python is the Makefile's $(PYTHON), else python3. It
# fails, saying why, when nothing is pinned or a pin or a version is not a plain number.
# CI runs the same target in the job a record would skip, and the two lines must match.
#
# THE BUDGET. The hook's run takes under a minute on a laptop, so it runs only the fast
# kinds: lint, format, typecheck, toolchain preflight. Suites, builds and services run on
# GitHub. A repo names its fast kinds in a `pre-push` target, which may only name targets
# its CI also runs: nothing runs only on the laptop. A repo whose whole `make check` runs
# under a minute may make `pre-push` just `check`, or have none.
#
# The hook runs `make pre-push` where the Makefile has that target and `make check` where
# it has not, or where a match-anything rule (`%:`) hides whether it has. A `pre-push` pass
# writes no stamp, so it skips nothing and publishes no record: only a full `make check`
# makes one (THE STAMP, THE PASS RECORD). The hook catches forgetting, not defiance:
# `git push --no-verify` skips it. It is written into this clone's own hooks directory,
# `git rev-parse --git-path hooks`, which in a linked worktree (where `.git` is a file)
# is the main clone's, so one install serves every worktree. It never writes global git
# config, and it refuses, exit 1, when `core.hooksPath` is set in a config file (git
# would read a different directory) or when a pre-push hook that is not this one is
# already there. A `core.hooksPath` set for one command only (`git -c`, GIT_CONFIG_*:
# the review desk's install points it at a throwaway directory) is that caller's choice:
# the install writes nothing and exits 0.
#
# GNU make 3.81 or later (macOS ships 3.81): `define` without an operator, target-specific
# `export`, `.DEFAULT_GOAL`.

# Including this file must not make `install-hooks` the repo's default goal.
_tb_install_hooks_goal := $(.DEFAULT_GOAL)

# The hook, as written to disk. `$$` is one `$` in the file.
define _tb_prepush_hook
#!/bin/sh
# tracebloc pre-push hook, written by `make install-hooks` (.github/install-hooks.mk).
# Runs the repo's fast `make pre-push` before a push, or `make check` where it has none,
# and publishes the pass record of each pushed tree a full `make check` passed (THE PASS
# RECORD in the snippet). Rewritten by every `make install-hooks`: change the snippet,
# never this file.

# A delete or no-op push carries no new commits (local sha all zeros): nothing to check.
z=0000000000000000000000000000000000000000
pushed=
while read -r _ local_sha _ _; do
  [ "$$local_sha" != "$$z" ] && pushed="$$pushed $$local_sha"
done
[ -z "$$pushed" ] && exit 0

# A GUI client may start hooks with a PATH that has no make, and may offer no --no-verify.
if ! command -v make >/dev/null 2>&1; then
  echo "pre-push: make is not on PATH, so make check did not run; the push proceeds."
  exit 0
fi

# Git exports GIT_DIR and friends into hooks; a nested git in a linked worktree then
# fails. Clear them so make check runs as it does from a shell.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX GIT_COMMON_DIR GIT_OBJECT_DIRECTORY

# The hook outlives the targets it names: only a branch with no check target passes unchecked.
if ! plan_out="$$(LC_ALL=C make -n check 2>&1)"; then
  case "$$plan_out" in
    *"No rule to make target 'check'."*|*"No rule to make target \`check'."*)
      echo "pre-push: this checkout has no check target, so nothing ran; the push proceeds."
      exit 0 ;;
  esac
  printf '%s\n' "$$plan_out"
  echo "pre-push: make cannot plan check (above), so the push is refused."
  exit 1
fi

# guard-toolchain, where a repo has one, says whether the tools check needs are on PATH.
if make -n guard-toolchain >/dev/null 2>&1; then
  if ! guard_out="$$(make guard-toolchain 2>&1)"; then
    printf '%s\n' "$$guard_out"
    echo "pre-push: the tools the checks need are missing (above), so nothing ran; the push proceeds."
    exit 0
  fi
fi

# Every pushed commit's tree already passed make check here (THE STAMP in the snippet).
$(_tb_check_stamps)
trees=
for sha in $$pushed; do
  tree="$$(git rev-parse --verify -q "$$sha^{tree}")"
  if [ ! -f "$$stamps/$$tree" ]; then
    trees=
    break
  fi
  trees="$$trees $$tree"
done
if [ -n "$$trees" ]; then
  echo "pre-push: make check already passed here on tree$$trees, so it did not run again."
elif pre_out="$$(LC_ALL=C make -n pre-push 2>&1)" && ! make -n tracebloc-no-such-goal >/dev/null 2>&1; then
  # The fast kinds only (THE BUDGET): no stamp, so this push publishes no record.
  make pre-push || exit $$?
  echo "pre-push: make pre-push passed; it leaves no pass record, so CI runs make check."
else
  # No pre-push target, or a match-anything rule (`%:`) that plans every goal and so hides
  # whether there is one, runs the full make check. A plan that fails otherwise refuses.
  case "$$pre_out" in
    *"No rule to make target 'pre-push'."*|*"No rule to make target \`pre-push'."*) ;;
    *)
      if ! make -n tracebloc-no-such-goal >/dev/null 2>&1; then
        printf '%s\n' "$$pre_out"
        echo "pre-push: make cannot plan pre-push (above), so the push is refused."
        exit 1
      fi ;;
  esac
$(_tb_check_stamped)
fi

$(_tb_publish_records)
exit 0
endef

# Where the stamps live; the hook reads them and `check-stamped` writes them.
define _tb_check_stamps
stamps="$$(git rev-parse --git-path tracebloc-check-stamps 2>/dev/null)"
endef

# `make check`, then the stamp of the tree it passed on (THE STAMP above). The hook ends
# with this and `check-stamped` runs it, so both write the same stamp.
define _tb_check_stamped
$(_tb_check_stamps)
clean_tree() {
  st="$$(git status --porcelain --untracked-files=all --ignore-submodules=none 2>/dev/null)" || return 1
  [ -z "$$st" ] && git rev-parse --verify -q 'HEAD^{tree}'
}
before="$$(clean_tree)"
make check || exit $$?
after="$$(clean_tree)"
if [ -z "$$before" ]; then
  echo "make check passed on a work tree that differs from HEAD, so no stamp was written."
elif [ "$$before" != "$$after" ]; then
  echo "make check passed, but the work tree changed while it ran, so no stamp was written."
elif mkdir -p "$$stamps" && : > "$$stamps/$$before"; then
  ls -t "$$stamps" | sed 1,20d | while read -r old; do rm -f "$$stamps/$$old"; done
  echo "make check passed on tree $$before; a push of it skips the pre-push run."
  if toolchain="$$(make -s --no-print-directory pass-record-toolchain)"; then
    printf '%s\n' "pass-record/1 tree=$$before toolchain=$$toolchain" > "$$stamps/$$before"
    echo "Its pass record: pass-record/1 tree=$$before toolchain=$$toolchain"
  else
    echo "It has no pass record: its toolchain cannot be told (above), so CI runs every check."
  fi
else
  echo "make check passed, but its stamp could not be written in '$$stamps'."
fi
endef

# THE PASS RECORD, published (above): the pushed commits whose tree's stamp holds a record,
# posted by a detached step once this push's git has exited. $$2 is the remote's URL.
define _tb_publish_records
case "$$2" in
  *github.com[:/]*) slug="$${2#*github.com[:/]}" ;;
  *) slug= ;;
esac
slug="$${slug%/}"
slug="$${slug%.git}"
case "$$slug" in
  */*/*|*[!A-Za-z0-9_./-]*|/*|*/|.*) slug= ;;
  */*) ;;
  *) slug= ;;
esac
posts=
if [ -n "$$slug" ] && command -v gh >/dev/null 2>&1; then
  for sha in $$pushed; do
    tree="$$(git rev-parse --verify -q "$$sha^{tree}")" || continue
    case "$$(sed -n 1p "$$stamps/$$tree" 2>/dev/null)" in
      "pass-record/1 tree=$$tree toolchain="?*) posts="$$posts $$sha" ;;
    esac
  done
fi
if [ -n "$$posts" ]; then
  parent=$$PPID
  (
    n=0
    while kill -0 "$$parent" 2>/dev/null && [ "$$n" -lt 120 ]; do
      sleep 1
      n=$$((n + 1))
    done
    for sha in $$posts; do
      tree="$$(git rev-parse --verify -q "$$sha^{tree}")" || continue
      record="$$(sed -n 1p "$$stamps/$$tree" 2>/dev/null)"
      gh api -X POST "repos/$$slug/statuses/$$sha" -f state=success -f context=tracebloc/pass-record -f "description=$$record" --silent
    done
  ) </dev/null >/dev/null 2>&1 &
  echo "pre-push: the pass record of$$posts goes to $$slug once this push lands."
fi
endef

# THE TOOLCHAIN (above), as one line on stdout; a reason on stderr and exit 1 when it
# cannot be told.
define _tb_toolchain
fp=
pin() {
  sed -n '1{s/^[[:space:]]*[vV]\{0,1\}//;s/[[:space:]]*$$//;p;}' "$$1"
}
add() {
  n="$$(printf '%s' "$$2" | tr -cd . | wc -c | tr -d ' ')"
  have="$$(printf '%s' "$$3" | cut -d. -f"1-$$((n + 1))")"
  case "$$2" in
    ''|.*|*.|*..*|*[!0-9.]*)
      echo "pass-record-toolchain: $$1 is pinned to '$$2', which is not a version number" >&2
      exit 1 ;;
  esac
  case "$$have" in
    ''|.*|*.|*..*|*[!0-9.]*)
      echo "pass-record-toolchain: the $$1 here reports '$$3', which is not a version number" >&2
      exit 1 ;;
  esac
  fp="$${fp:+$$fp,}$$1=$$have"
}
if [ -f .python-version ]; then
  add python "$$(pin .python-version)" "$$($(or $(PYTHON),python3) -c 'import platform; print(platform.python_version())' 2>/dev/null)"
fi
if [ -f .nvmrc ]; then
  add node "$$(pin .nvmrc)" "$$(node -p process.versions.node 2>/dev/null)"
elif [ -f .node-version ]; then
  add node "$$(pin .node-version)" "$$(node -p process.versions.node 2>/dev/null)"
fi
if [ -f go.mod ]; then
  add go "$$(sed -n 's/^go[[:space:]][[:space:]]*\([0-9][0-9.]*\)[[:space:]]*$$/\1/p' go.mod | sed -n 1p)" \
    "$$(go env GOVERSION 2>/dev/null | sed 's/^go//')"
fi
if [ -z "$$fp" ]; then
  echo "pass-record-toolchain: this checkout pins no runtime in a version file, so there is no toolchain to record" >&2
  exit 1
fi
echo "$$fp"
endef

.PHONY: install-hooks hooks-status check-stamped pass-record-toolchain

install-hooks: export TRACEBLOC_PREPUSH_HOOK = $(_tb_prepush_hook)
install-hooks:
	@if ! git rev-parse --git-dir >/dev/null 2>&1; then \
	  echo "install-hooks: not a git checkout, so there is no hook to install"; \
	  exit 0; \
	fi; \
	hp="$$(git config --get core.hooksPath || true)"; \
	if [ -n "$$hp" ] && [ "$$(git config --show-scope --get core.hooksPath 2>/dev/null | cut -f1)" = command ]; then \
	  echo "install-hooks: core.hooksPath is '$$hp' for this command alone (git -c or GIT_CONFIG_*), so its caller chose where hooks go; nothing installed"; \
	  exit 0; \
	fi; \
	if [ -n "$$hp" ]; then \
	  echo "install-hooks: core.hooksPath is set to '$$hp', so git would not read a hook installed in this clone." >&2; \
	  echo "  Unset it (git config --unset core.hooksPath, with --global if it is set there), then run make install-hooks again." >&2; \
	  exit 1; \
	fi; \
	hook="$$(git rev-parse --git-path hooks)/pre-push"; \
	if [ -e "$$hook" ] && ! grep -q 'tracebloc pre-push hook' "$$hook"; then \
	  echo "install-hooks: $$hook is not this repo's pre-push hook, so it is left as it is." >&2; \
	  echo "  Move what it does elsewhere and delete it, then run make install-hooks again." >&2; \
	  exit 1; \
	fi; \
	mkdir -p "$$(dirname "$$hook")" && \
	printf '%s\n' "$$TRACEBLOC_PREPUSH_HOOK" > "$$hook.tmp.$$$$" && \
	chmod +x "$$hook.tmp.$$$$" && \
	mv -f "$$hook.tmp.$$$$" "$$hook" && \
	echo "install-hooks: $$hook installed; make pre-push, or make check where there is none, runs before every push"

hooks-status: export TRACEBLOC_PREPUSH_HOOK = $(_tb_prepush_hook)
hooks-status:
	@hook="$$(git rev-parse --git-path hooks)/pre-push" || exit 1; \
	hp="$$(git config --get core.hooksPath || true)"; \
	if [ -x "$$hook" ] && [ "$$(cat "$$hook")" = "$$TRACEBLOC_PREPUSH_HOOK" ]; then \
	  echo "hooks-status: current: $$hook"; \
	else \
	  echo "hooks-status: missing or stale: $$hook$${hp:+ (core.hooksPath is set to $$hp)}; run make install-hooks" >&2; \
	  exit 1; \
	fi

check-stamped: export _TB_CHECK_STAMPED = $(_tb_check_stamped)
check-stamped:
	@sh -c "$$_TB_CHECK_STAMPED" check-stamped

pass-record-toolchain: export _TB_TOOLCHAIN = $(_tb_toolchain)
pass-record-toolchain:
	@sh -c "$$_TB_TOOLCHAIN" pass-record-toolchain

.DEFAULT_GOAL := $(_tb_install_hooks_goal)
