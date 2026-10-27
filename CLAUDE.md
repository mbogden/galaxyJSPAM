# galaxyJSPAM: agent instructions

Rules for AI agents working in this repo.

## What this branch is
`modern` is a from-scratch rewrite of galaxyJSPAM (SPAM galaxy-collision simulations,
scoring and optimization) as an installable, reproducible package.

- `working_research` holds the old code and is still used for research until `modern` reaches parity.
  Port code from it one module at a time; don't copy whole folders.
- Any science change committed to `working_research` after this branch was created must be ported here too.

## Rules
- Ask before `git push`, deleting files, or any force/reset operation.
- Never commit data or large outputs. Data lives outside the repo (`../data/` in the workspace).
- Keep commits small, with one change each, and explain why in the message.
- Add a dependency only when code needs it.
- Python 3.12. Container and cluster work target CUDA 12.x (the cluster driver is 12.4) and OpenMPI 4.1.6.

## Planned layout
To be filled in as the package takes shape: `pyproject.toml` + `uv`, `src/` layout, Fortran built with meson-python, tests.
