#!/usr/bin/env python
"""Verify a brachiation scene asset before training on it (pure MuJoCo, CPU).

Checks the things that actually decide whether a run is comparable to the earlier
ones, in the order that matters:

1. **geometry provenance** - bar spacing, ``nq``/``ngeom``/``nkey``, and the hang
   keyframe itself.  For the M3.1 easy-geometry asset (``full035``) the keyframe
   and ``bar_x[0]`` must be *identical* to ``full``: the hang pose is solved from
   B0, which does not move, so the only change is that the next bar is closer;
2. **reachability** - the static right-arm IK limit to B1 (M3.0's number).  This
   is the quantity that turned "graze" into "grasp" territory: 0.031 m at 0.40 vs
   0.0007 m at 0.35;
3. **hang stability** - hold the keyframe for 1 s and report the drift, i.e. the
   M1 acceptance behaviour on the new asset.

Usage::

    .venv-warp/bin/python src/check_scene.py --scene full035
    .venv-warp/bin/python src/check_scene.py --scene full035 --ref full
    .venv-warp/bin/python src/check_scene.py --scene full --no-ik     # fast

``--ref`` runs the same checks on another registry scene so the two can be
diffed line by line (it is how the numbers in docs/M2_JaxGCRL接入记录.md 9.32
were produced).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import m3_swing as m3  # noqa: E402  (pure MuJoCo; importing the env would pull in jax)

PITCH = np.linspace(-3.0, 1.0, 41)
ROLL = np.linspace(-2.0, 2.0, 21)
ELBOW = np.linspace(0.0, 2.5, 21)
HIP = "left_hip_pitch_joint"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="full035",
                    help="registry name (full/full035) or a path to an exported XML")
    ap.add_argument("--ref", default=None, help="optional second scene to compare against")
    ap.add_argument("--no-ik", action="store_true", help="skip the static IK sweep (~10 s)")
    ap.add_argument("--hold", type=float, default=1.0, help="keyframe hold seconds")
    args = ap.parse_args()

    names = [args.scene] + ([args.ref] if args.ref else [])
    rows = []
    for name in names:
        path = m3.SCENES.get(name, name)
        rig = m3.Rig(scene=path)
        m, d = rig.m, rig.d
        kq = rig.key_qpos
        hip = m.jnt_qposadr[m3.joint_id(m, HIP)]
        rig.reset()

        # static kinematics: |p_L - B0| and |p_R - B1| at the keyframe, then the sweep
        gap_hang = rig.dist_to_bar("right", 1)
        gap_left = rig.dist_to_bar("left", 0)
        ik_best = None if args.no_ik else m3.ik_reach(rig, PITCH, ROLL, ELBOW)["d"]

        # M1 acceptance: hold the keyframe ctrl and see how far it drifts
        rig.reset()
        d.ctrl[:] = rig.key_ctrl
        z0, x0 = float(d.qpos[2]), float(d.qpos[0])
        for _ in range(int(round(args.hold / m.opt.timestep))):
            m3.mujoco.mj_step(m, d)
        rows.append(dict(
            name=name, path=os.path.basename(path), spacing=rig.bar_x[1] - rig.bar_x[0],
            bar_x=rig.bar_x, nq=m.nq, ngeom=m.ngeom, nkey=m.nkey,
            key_z=float(kq[2]), key_hip=float(kq[hip]), gap_left=gap_left,
            gap_hang=gap_hang, ik=ik_best,
            dz=float(d.qpos[2]) - z0, dx=float(d.qpos[0]) - x0,
            fell=bool(d.qpos[2] < rig.bar_z - m3.FALL_DEPTH),
        ))

    for r in rows:
        print(f"== {r['name']}  ({r['path']})")
        print(f"   bars          : spacing={r['spacing']:.4f} m  "
              f"bar_x={np.round(r['bar_x'], 4).tolist()}")
        print(f"   model         : nq={r['nq']} ngeom={r['ngeom']} nkey={r['nkey']}")
        print(f"   hang keyframe : z={r['key_z']:.4f}  hip_pitch={r['key_hip']:+.4f}")
        print(f"   static @key   : |p_L-B0|={r['gap_left']:.4f}  |p_R-B1|={r['gap_hang']:.4f}")
        print(f"   static IK lim : " + ("skipped" if r["ik"] is None else f"{r['ik']:.4f} m"))
        print(f"   hold {args.hold:g}s      : dz={r['dz'] * 1e3:+.1f} mm  dx={r['dx'] * 1e3:+.1f} mm  "
              f"fell={r['fell']}")

    if len(rows) == 2:
        a, b = rows
        same_key = (abs(a["key_z"] - b["key_z"]) < 1e-9
                    and abs(a["key_hip"] - b["key_hip"]) < 1e-9
                    and abs(a["bar_x"][0] - b["bar_x"][0]) < 1e-9)
        print(f"\n[single-variable check] keyframe + bar0 identical: {same_key}  "
              f"(bar gap {a['spacing']:.3f} -> {b['spacing']:.3f} m)")
        if not same_key:
            print("WARN the two scenes differ in the keyframe/B0, so a run on one is "
                  "NOT a single-variable change from the other")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
