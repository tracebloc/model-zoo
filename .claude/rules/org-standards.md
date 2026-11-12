<!-- Synced from the private source repo by scripts/standards-sync.py. Never edit by hand:
     change the canon there and the next sync rewrites this file. -->

## tracebloc engineering standards (org-wide)

A digest, one line per rule, of the binding org-standards.md. Read its section before acting on a line. Never edit it here.

### Where the rules and the automation live

- Org automation lives in `the private source repo`; per-repo callers pin `@main`; nothing new goes to `tracebloc/.github`.
- Mirrors take no work: no PR, branch or internal issue on a `mirror_of:` row; work in its source.

### Branches & PRs

- On the train, branch off and target `develop`, never `staging` or `main`.
- Off it, `repo-inventory.yml` says the model; never create a missing `develop`.
- `git fetch`, branch from the tip of `develop`, one author, no stacking; a day-old branch merges `develop` before review.
- One self-contained change per PR, split at 1000+ lines; refactors apart from behaviour.
- A contract change opens its consumers' updates; all merge back to back, every `develop` green.
- Clean up your branches (`git reap`): those whose PR you opened (`scripts/branch_owner.py`).
- Branches `feat/ fix/ docs/ sec/ ci/ chore/ test/` + issue + slug; commits `type(scope): summary`; `(scope)` is the component, never the ticket number.
- A title number needs `Closes <owner>/<repo>#N` (done) or `Part of <owner>/<repo>#N` (partial).
- Assign yourself; `tracebloc-review` is the only required reviewer; add a human only by name.
- Bodies carry `Release note:`, `Surface:` (customer voice, or `none` by decision) and the Billing box.
- A PR ships alone: additive, dark behind a default-off flag, or a draft until its need is in Prod.
- `Merge after: tracebloc/<repo>#<n>` orders merges into `develop`, never a release, so it takes no ` on staging`; a PR that needs another PR in Prod is a draft, and it is one before review.
- A human named as reviewer answers within one business day.
- A `claude-*` label is a session at work: never push to, re-request review on or merge its PR.

### Quality bar

- Before every push, lint and test your change; never push a branch you believe is red.
- Read the full diff before opening the PR: you own every line.
- End with evidence, the check's output; an unverified change does not ship.
- Fix the class: other call sites and other inputs to the guard, or say it cannot be listed.
- Where the desk runs, never poll CI or Bugbot; elsewhere triage your PR. Fix or answer every finding.
- A Bugbot finding measured unreachable takes this path:
<!-- unreachable-finding-path:begin -->
  1. **Measure and post.** Construct the input the finding needs, run it, and post the input, the command and the result on its thread.
  2. **Pin it in the same PR.** The PR that answers it adds a test or guard that goes red once an input reaches the code; the thread names it.
  3. **Second reader.** The review account, never whoever measured, checks both and resolves the thread; missing either, it stays a finding.
  4. **Close.** The pinning PR `Closes` any ticket filed for the finding, and its opener posts one comment there whose first line is exactly `Closed: unreachable, pinned by <owner>/<repo>#<number>`, naming the PR. That marker counts only on a ticket the named PR closed. It tells; it asks nobody.
<!-- unreachable-finding-path:end -->
- A recurring finding becomes a `.cursor/BUGBOT.md` rule; style rules live in tooling, not prose.
- Never commit secrets, tokens or customer data; keep internal refs out of public PR text.
- A guard reads the real thing and fails closed: derive, never restate; "cannot tell" is a finding.
- A guard is proven by breaking it: assert the mutation applied; for an enum, alias or vocabulary, derive the input domain from the producer's declared surface and test all of it; arm while green; a red nightly is a bug.
- Comments give the current reason once, in three lines or fewer; no history, line numbers or "today".
- Deferred work is `TODO(tracebloc/<repo>#N)`; edit comments, never append; reviewers hold that line.

### Engineer kanban

- Every ticket has a `Status`, new ones `Backlog`; a `work-type:bug` goes straight to `Ready`.
- `off-board` is only for an issue that is a ledger or a bot's record.
- Take from `Ready` first; a `Backlog` pull is normal: say what you take.
- No dev-side review: FR is once, on staging, by `/fr-pass` or `Ready for prod`; self-signoff counts.
- `fr-gate` blocks: fix the board or the work; exit 3: re-run it. `skip-fr-gate` is for emergencies.

### Releases & publishing

- The release train is the only path to `staging`, `main` and registries; bump a version only to `X.Y.0`.
- Fix a promotion PR's findings on its source branch, never on the PR.

### Filing issues

- Internal work (plans, epics, security, infra, any customer) goes in `(internal ref)`, never a public repo. When in doubt: `(internal ref)`.
- A public repo gets only issues a stranger could act on; its PR text names nothing private.

### Product words

- Every user-facing word comes from `TERMINOLOGY.md` in docs; a missing word goes there first; never copy its table into another file.
- With a copy catalog, edit the golden first; never regenerate another's golden to go green.

### AI-assisted sessions (Claude Code, etc.)

- A session never merges, closes another's PR, deletes another's branch or force-pushes unless told.
- Functional sign-off is a person's: a session never comments `/fr-pass` or removes `fr-hold`.
- New guidance goes to the narrowest file that loads when needed: `<dir>/CLAUDE.md`, or `.claude/rules/<topic>.md` with `paths:`; reference to `docs/`; history to the commit message. The root changes only for a new command or a rule every session needs.
- If your change makes a CLAUDE.md, `.claude/rules/` file, BUGBOT.md or runbook false, fix it in the same PR.
