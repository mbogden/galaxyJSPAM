# galactic_tidal_constraints: agent instructions

Rules for AI agents working in this repo.

## What this branch is
`modern` is a from-scratch rewrite of this project (formerly galaxyJSPAM: SPAM
galaxy-collision simulations, scoring and optimization) as an installable, reproducible package.

- `working_research` holds the old code and is still used for research until `modern` reaches parity.
  Port code from it one module at a time; don't copy whole folders.
- Any science change committed to `working_research` after this branch was created must be ported here too.

## Rules
- Ask before `git push`, deleting files, or any force/reset operation.
- Never commit data or large outputs. Data lives outside the repo (`../data/` in the workspace).
- Keep commits small, with one change each, and explain why in the message.
- Add a dependency only when code needs it.
- Follow the decisions in `PLAN.md`. Don't add tools or structure that aren't listed there without asking.
