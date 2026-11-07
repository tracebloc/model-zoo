# Bugbot guide

What Bugbot should know before reviewing a diff in THIS repo: the stack, the
conventions that are not enforced by a linter, and the defect classes that
have recurred here. Per-repo by design -- any sibling repo's `.cursor/BUGBOT.md`
shows the shape. Presence is asserted by the conformance audit
(repo-inventory.yml `quality_files`); content is yours.

<!-- comment-policy:begin -->
**Comments** (code comments and docstrings)

1. **State the current reason, in three lines or fewer.** A comment records a
   constraint, hazard, invariant or contract the code cannot show. History goes in the
   commit message: what it used to do, review rounds, reviewer or bot names, "this PR",
   "the first version", incident arithmetic. A contract docstring (parameters, units,
   raises) may be longer; it must not restate the signature.
2. **One canonical site per rationale.** Put it on the symbol that owns the decision.
   Every other site gets a one-line pointer to that symbol (`# NULL semantics: see
   CycleInformation.size_bytes`), never a copy.
3. **Nothing that goes stale silently.** No line numbers, no `file.py:NNN`, no counts of
   call sites or instances, no "today", "currently", "for now", "this week". Name the
   symbol. Cite a ticket only as a TODO (rule 4) or as a pointer to where a decision is
   argued (an RFC); never the ticket that fixed something, because `git blame` finds it.
   A regression test may name the bug it pins.
4. **Deferred work is `TODO(tracebloc/<repo>#N)`**, with an open ticket. No `FIXME`,
   no `XXX`, no "left to a follow-up". In a public repo the ticket is in that repo.
5. **Edit, don't append.** When you change code under a comment, rewrite that comment to
   match. Answering a review finding means changing the code and, if needed, the existing
   comment's reason — never adding a block that records the finding.
6. **Reviewers hold the same line.** Do not ask for a comment that explains a fix or
   credits a finding; ask for the reason, or for the commit message. Flag a comment the
   diff made false.

**PR bodies** (the description, read beside the diff)

- **A PR body that names another PR it needs before it can ship.** A PR ships to prod
  alone, in any order with every other PR; one that needs another stays a draft until
  that PR is in prod. On a PR that is not a draft, flag each body line, outside HTML
  comments, that makes it wait for another PR or issue: a `PAIR:` line, a `SHIPS-AFTER`
  line, "ships after", "depends on" or "needs" a named PR, or a `Merge after:` line
  ending in ` on staging`. A plain `Merge after: <owner>/<repo>#<n>` line is no finding:
  it orders merges into `develop`. Say which line, and that a ticked `Can this ship
  alone?` box beside it contradicts it. The answer is the author's: split the change,
  make this PR a draft, or reword a line that only gives context. Low severity and a
  flag only: never request changes for it, and never treat it as a reason to hold the PR.
<!-- comment-policy:end -->
