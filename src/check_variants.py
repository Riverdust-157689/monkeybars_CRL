"""Static-ish guard: every goal variant must be self-consistent.

Catches a bug class that slipped through ``check_args.py`` / ``check_metrics.py``:
they never call ``_achieved_goal`` / ``_state_features`` with their *default*
arguments, so a variant whose extra goal feature has the wrong shape (or is only
correct when the caller passes the real value) passes both guards and only blows
up later -- in ``render_policy.py``, or at the first eval of some other variant.

Checks, for every entry of ``GOAL_VARIANTS``:
  * ``_achieved_goal(data)`` and ``_state_features(data, 0)`` called with DEFAULTS
    produce exactly ``len(goal_indices)`` / ``state_dim`` finite floats;
  * ``goal_set`` has one column per goal entry (one row for bar-independent goals,
    ``n_bars`` rows otherwise);
  * ``observation_size == state_dim + len(goal_indices)``.

With ``--deep`` it additionally runs 3 real ``step``s per variant and asserts that
``_achieved_goal_info(state.pipeline_state, state.info)`` — the SINGLE entry point
``render_policy.py`` now uses — reproduces ``state.metrics["dist"]`` exactly.  That
is the invariant that broke silently before: the render side hand-assembled the
per-variant history arguments, and one variant's `hnext` was dropped, so the
rendered goal distance was computed from a different state than the one training
used.  (Cheap enough to opt into before a long run; skipped by default because it
compiles a `step` per variant.)

Usage:  .venv-warp/bin/python src/check_variants.py [--scene full035] [--deep]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "third_party", "jaxgcrl"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="full035",
                    choices=["full", "full035", "mesh", "allprim"])
    ap.add_argument("--deep", action="store_true",
                    help="also run 3 real steps per variant and check that "
                         "_achieved_goal_info reproduces metrics['dist']")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from envs.brachiation import GOAL_VARIANTS, Brachiation

    scene = os.path.join(REPO, "assets", "g1_brachiation",
                         {"full": "scene_bars.xml", "full035": "scene_bars_d035.xml",
                          "mesh": "scene_bars_mjx.xml",
                          "allprim": "scene_bars_mjx_allprim.xml"}[args.scene])
    bad = 0
    print("%-14s %6s %6s %6s %6s  %s" % ("variant", "state", "goal", "obs", "rows", "status"))
    for v in GOAL_VARIANTS:
        try:
            env = Brachiation(scene_xml=scene, impl="warp", goal_variant=v)
            ach = np.asarray(env._achieved_goal(env._data0))          # defaults
            st = np.asarray(env._state_features(env._data0, jnp.zeros(14)))
            gs = np.asarray(env.goal_set)
            ok = (ach.shape == (len(env.goal_indices),)
                  and st.shape == (env.state_dim,)
                  and gs.shape[1] == len(env.goal_indices)
                  and gs.shape[0] in (1, env.n_bars)
                  and np.isfinite(ach).all() and np.isfinite(st).all()
                  and env.observation_size == env.state_dim + len(env.goal_indices))
            print("%-14s %6d %6d %6d %6d  %s" % (
                v, st.shape[0], len(env.goal_indices), env.observation_size, gs.shape[0],
                "OK" if ok else "MISMATCH"))
            bad += 0 if ok else 1
            if ok and args.deep:
                # the render/metrics agreement invariant (see the module docstring)
                state = env.reset(jax.random.PRNGKey(0))
                act = jnp.zeros(env.action_size)
                for _ in range(3):
                    state = env.step(state, act)
                ag = np.asarray(env._achieved_goal_info(state.pipeline_state, state.info))
                d_ref = float(state.metrics["dist"])
                d_got = float(np.linalg.norm(ag - np.asarray(state.info["goal"])))
                fine = ag.shape == (len(env.goal_indices),) and np.isfinite(ag).all()
                agree = abs(d_ref - d_got) < 1e-4
                print("%-14s %6s %6s %6s %6s  deep %s (dist %.6f vs %.6f)" % (
                    v, "-", "-", "-", "-",
                    "OK" if (fine and agree) else "MISMATCH",
                    d_ref, d_got))
                bad += 0 if (fine and agree) else 1
        except Exception as exc:                      # noqa: BLE001
            print("%-14s %6s %6s %6s %6s  FAIL %s: %s" % (
                v, "-", "-", "-", "-", type(exc).__name__, str(exc)[:60]))
            bad += 1
    if bad:
        print("FAIL %d variant(s) inconsistent" % bad)
        return 1
    print("OK  all %d variants are self-consistent" % len(GOAL_VARIANTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
