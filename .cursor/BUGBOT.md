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
<!-- comment-policy:end -->
