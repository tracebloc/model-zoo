## Summary
<!-- 1–3 sentences. What does this PR do and why? -->

## Related
<!-- Closes #123 / Ref tracebloc/other-repo#456 -->

## Release note
<!-- REQUIRED on `feat` and `fix` pull requests in a customer-facing repository, where the `release-note / check` context reads it; optional everywhere else. ONE line in the customer's voice -- what a user notices, never what the code does -- or the literal `none`. `Surface:` is one of platform / client / sdk / cli / docs. Notes are compiled into the product changelog when a release is cut; titles never are. The values are blank on purpose: writing `none` is a decision, not a default. -->
Release note:
Surface:

## Type of change
- [ ] Feature
- [ ] Bug fix
- [ ] Tech-debt / refactor
- [ ] Docs
- [ ] Security / hardening
- [ ] Billing / metering / numerics
- [ ] Breaking change

## Test plan
<!-- What did you test? Commands run? Manual steps? -->

## Screenshots / recordings
<!-- For UI changes. Remove if N/A. -->

## Deployment notes
<!-- Env vars, migrations, feature flags. Remove if N/A. -->

## Checklist
- [ ] Tests added / updated and passing locally
- [ ] Docs updated if behavior or config changed
- [ ] No secrets / credentials in the diff
- [ ] For security-sensitive paths: appropriate reviewer requested
- [ ] Can this ship alone? It ships to Prod in any order with every other PR: additive (expand, then contract) or dark behind a flag that is off by default. If not, split the change: each part that can ships alone, and a part that needs another PR stays a draft until that PR is in Prod. Unticked is a draft or a two-sided break. A `Merge after:` line may order merges into `develop`, never wait on another PR's release.
