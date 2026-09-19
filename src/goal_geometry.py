"""Calibrate the goal-distance thresholds against the actual goal geometry.

``Brachiation`` judges success by ``||achieved_goal - goal|| < goal_reach_thresh``
in the 10-dim goal space ``[x_torso, z_torso, p_L(3), p_R(3), c_L, c_R]``.  This
script prints the canonical hang poses of each bar and their pairwise distances, so
the threshold can be read as "*fraction of a bar*" instead of a magic number.

It needs **no simulation** -- the canonical goals are built from the hang keyframe
at import time -- so it runs in a couple of seconds even on CPU::

    .venv-warp/bin/python src/goal_geometry.py
    .venv-warp/bin/python src/goal_geometry.py --scene full --thresh 0.35
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, os.path.join(HERE, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mjx_backend as mb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="jax", choices=["jax", "warp"])
    ap.add_argument("--scene", default="allprim", choices=["full", "full035", "mesh", "allprim"])
    ap.add_argument("--thresh", type=float, default=0.35, help="goal_reach_thresh to interpret")
    args = ap.parse_args()

    if args.impl == "warp":
        mb.enable_warp_compat()
    from envs.brachiation import create_brachiation

    env = create_brachiation(impl=args.impl, scene=args.scene)
    g = np.array(env.goal_set)
    np.set_printoptions(precision=4, suppress=True)
    print(f"bar_spacing={env.bar_spacing:.4f} m  n_bars={env.n_bars}  "
          f"goal_dim={g.shape[1]}")
    print("goal layout: [x_torso, z_torso, p_L(3), p_R(3), c_L, c_R]")
    for i in range(len(g)):
        print(f"  B{i}: x={g[i,0]:+.4f} z={g[i,1]:+.4f} pL={g[i,2:5]} pR={g[i,5:8]} "
              f"c={g[i,8:10]}")

    D = np.linalg.norm(g[:, None, :] - g[None, :, :], axis=-1)
    print("\npairwise ||goal_i - goal_j|| (rows = state hanging on B_i):")
    print(D)

    step = float(D[0, 1]) if len(g) > 1 else float("nan")
    print(f"\none-bar step  = {step:.4f} m   (= sqrt(3) * bar_spacing = "
          f"{np.sqrt(3)*env.bar_spacing:.4f})")
    print(f"threshold {args.thresh:.3f} = {args.thresh/step:.3f} of a bar")
    for k in range(len(g)):
        d = float(D[k, len(g) - 1])
        tag = "success" if d < args.thresh else (
            "success_easy" if d < 3 * args.thresh else "-")
        print(f"  from B{k} to goal B{len(g)-1}: dist={d:.4f}  ({d/step:.2f} bars)  {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
