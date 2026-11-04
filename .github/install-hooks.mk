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
#
# THE STAMP. A pass of `make check` is recorded as HEAD's tree hash, an empty file in
# `git rev-parse --git-path tracebloc-check-stamps` (this checkout's git directory, never
# the work tree), and only when make check exited 0 and `git status` showed nothing
# tracked or untracked (not ignored) differing from HEAD, before and after the run, on
# the same HEAD tree. The newest 20 stamps are kept. The hook skips `make check` only
# when the tree of every commit it is pushing has a stamp; a tree it cannot read runs it.
# The hook's own passing run stamps too, so a retried push is free.
#
# The hook runs `make check` and nothing else. It catches forgetting, not defiance:
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
# Runs `make check` before a push, so a push that would be red in CI is caught here
# first. Rewritten by every `make install-hooks`: change the snippet, never this file.

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
    echo "pre-push: the tools make check needs are missing (above), so it did not run; the push proceeds."
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
  exit 0
fi

$(_tb_check_stamped)
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
else
  echo "make check passed, but its stamp could not be written in '$$stamps'."
fi
exit 0
endef

.PHONY: install-hooks hooks-status check-stamped

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
	echo "install-hooks: $$hook installed; make check runs before every push"

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

.DEFAULT_GOAL := $(_tb_install_hooks_goal)
