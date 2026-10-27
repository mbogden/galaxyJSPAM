# Modernization Plan

The `modern` branch is a rewrite of galaxyJSPAM as an installable, reproducible package.
This file records what has been decided. Items are added one at a time as they are decided.

## Scope
- Start by building around TNG → SPAM: simulate and optimize, for the TNG vs SPAM paper.
- Port only what that paper needs. The scope grows with the code. Image creation and
  machine scoring come later.

## Decided
- **Packaging:** `pyproject.toml` + `uv`. The lockfile is part of reproducibility.
- **Environment:** one Docker image, run with Apptainer on the cluster.
  - CUDA 12.x only (the cluster driver is 12.4)
  - Ubuntu 24.04
  - OpenMPI 4.1.6, matching the cluster
  - Built locally, pushed to Docker Hub, pulled on the cluster with `apptainer pull`
- **Data:** never in git; it lives outside the repo.
- **Input:** command-line arguments with `argparse`.
- **Logging:** Python's standard `logging` module, used throughout.
  - DEBUG logs trace what the code did; checks that a result is correct belong in tests.
  - DEBUG statements are kept for development and testing. They must cost almost nothing when
    DEBUG is off:
    - Lazy formatting: `logger.debug("x = %s", x)`, not f-strings or `%` inside log calls.
    - No logging inside tight loops (per particle, per time step); log summaries per run or
      per generation.
  - One logger per module: `logger = logging.getLogger(__name__)`. Modules never configure logging.
  - Configure once at the entry point: the level comes from `--log-level`, and the format
    includes time, level, module, function and MPI rank. Quiet noisy libraries individually
    (e.g. `logging.getLogger("matplotlib").setLevel(logging.WARNING)`).
  - One log call per error: `logger.exception("...")` records the message and traceback,
    then raise. Don't log the same error again further up.

## Branches
- `modern`: created from `working_research`. The first commit cleared the old tree, so the
  old history is still reachable.
- `working_research`: stays in use for research until `modern` reaches parity.
- Old branches (`Master`, `old_master`) are kept as archive tags. Deleting them is not required.
- **At parity:** `modern` becomes `main`, `working_research` becomes `legacy`, and the GitHub
  default branch becomes `main`.

## To port
Science changes committed to `working_research` after `modern` was created
(2026-10-27, from `9f87fea`). Each one must also be made on `modern`.

| Date | Commit | Change | Ported? |
|---|---|---|---|
