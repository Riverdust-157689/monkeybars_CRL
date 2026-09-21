# jaxgcrl — vendored upstream sources (provenance)

This directory is **not** a git checkout.  It is the upstream Python package
[`MichalBortkiewicz/JaxGCRL`](https://github.com/MichalBortkiewicz/JaxGCRL) at

    commit 5a6e7a0   ("Fix cosine energy_fn: normalize per-sample, ...")

with this project's modifications applied, vendored so that a fresh `git clone`
of *this* repo needs no network access and no patch step to train.

## What is here

| kept | why |
|---|---|
| `jaxgcrl/**/*.py` (396 KB, 42 files) | everything this project imports: `jaxgcrl.agents.crl`, `jaxgcrl.utils`, plus the package `__init__`s |
| `LICENSE` | upstream licence (MIT) |
| `PROVENANCE.md` | this file |

## What is deliberately omitted

| omitted | size | why |
|---|---|---|
| `jaxgcrl/envs/assets/` | 36 MB | upstream's own benchmark meshes (Franka etc.); this project's physics lives in `src/envs/brachiation.py` + `assets/g1_brachiation/`, and upstream's envs are never instantiated |
| every other top-level file (`docs/`, `notebooks/`, `scripts/`, `tests/`, `imgs/`, `renders/`, `run.py`, `uv.lock`, `environment.yml`, ...) | ~20 MB | not used by this project |
| `.git`, `__pycache__`, `*.egg-info` | — | not source |

## The modification (the only delta from upstream)

`patches/jaxgcrl_crl_losses.patch` — 128 lines touching exactly two files:

* `jaxgcrl/agents/crl/crl.py`: `--entropy-param` (target entropy = -param x action_size,
  upstream hardcodes 0.5), `--expl-hold` (temporally correlated exploration noise),
  the assertion on `params` relaxed to a warning.
* `jaxgcrl/utils/evaluator.py`: the brachiation metric names added to the aggregator
  whitelist (without this, our `cov_*` / `advance_max` / `dual_*` metrics are silently
  dropped from `progress.csv`).

## How to verify / re-sync

```bash
# 1. does the vendored tree still equal upstream@5a6e7a0 + the patch?
tmp=$(mktemp -d)
git clone https://github.com/MichalBortkiewicz/JaxGCRL "$tmp/j"
git -C "$tmp/j" checkout 5a6e7a0
git -C "$tmp/j" apply "$PWD/patches/jaxgcrl_crl_losses.patch"
diff -r --exclude=assets --exclude=__pycache__ --exclude=.git "$tmp/j/jaxgcrl" third_party/jaxgcrl/jaxgcrl
rm -rf "$tmp"

# 2. after editing the vendored copy by hand, regenerate the patch from upstream.
#    This needs upstream's git object database, which is gitignored (it only exists
#    on the machine that vendored this tree).  On a fresh clone either keep a real
#    checkout (below) or restore just the object database:
#        tools/make_patch.sh --restore-upstream   # clones upstream at 5a6e7a0
#        tools/make_patch.sh --check              # must print byte-identical
```

If you would rather keep a real checkout (e.g. to rebase onto a newer upstream),
delete this directory and run `tools/setup_env.sh`, which clones upstream at the
pinned commit and applies the patch.  A **fork** of JaxGCRL is only worth it if
you intend to send changes upstream; for pinning *our* delta the patch file is
simpler and stays reviewable.
