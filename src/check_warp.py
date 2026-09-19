"""Is MJX-Warp usable? — end-to-end verification on one environment.

Checks, in order:
  1. Warp initialises and reports its devices.
  2. `mjx.put_model(impl="warp")` + `make_data` succeed on the chosen scene.
  3. A jitted `mjx.step` runs (first call includes kernel compilation).
  4. The `hang` keyframe is *maintained*: step N times and compare the pelvis
     trajectory with native MuJoCo on the same model.
  5. Report single-env steps/s (CPU numbers are indicative only; the real
     throughput gate is on the GPU machine).

Usage::

    .venv-warp/bin/python src/check_warp.py                      # mesh scene, warp
    .venv-warp/bin/python src/check_warp.py --scene allprim      # all-primitive scene
    .venv-warp/bin/python src/check_warp.py --seconds 2 --impl jax
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import mjx_backend as mb  # noqa: E402

SCENES = {
    "mesh": os.path.join(REPO, "assets", "g1_brachiation", "scene_bars_mjx.xml"),
    "allprim": os.path.join(REPO, "assets", "g1_brachiation", "scene_bars_mjx_allprim.xml"),
    "full": os.path.join(REPO, "assets", "g1_brachiation", "scene_bars.xml"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="mesh", choices=list(SCENES))
    ap.add_argument("--impl", default="warp", choices=["jax", "warp"])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--naconmax", type=int, default=8192)
    ap.add_argument("--njmax", type=int, default=2048)
    args = ap.parse_args()

    xml = SCENES[args.scene]
    print(f"[1] scene={args.scene} ({os.path.basename(xml)})  impl={args.impl}")
    if args.impl == "warp":
        ok = mb.enable_warp_compat()
        print(f"    warp importable: {ok}")
        if ok:
            print(f"    warp devices  : {mb.warp_devices()}")

    import jax
    import mujoco

    t0 = time.time()
    mj_model, mx_model, mx_data = mb.load_mjx(xml, impl=args.impl,
                                              naconmax=args.naconmax, njmax=args.njmax)
    print(f"[2] put_model + make_data OK  ({time.time()-t0:.1f}s)  nq={mj_model.nq} nv={mj_model.nv} nu={mj_model.nu}")

    mx_data = mb.reset_mjx_data(mj_model, mx_data)
    from mujoco import mjx as _mjx
    step = jax.jit(lambda d: _mjx.step(mx_model, d))

    t0 = time.time()
    mx_data = step(mx_data)
    jax.block_until_ready(mx_data.qpos)
    print(f"[3] first jitted step OK (kernel compile)  {time.time()-t0:.1f}s")

    n = max(1, int(args.seconds / mj_model.opt.timestep))
    traj = [np.array(mx_data.qpos)]
    t0 = time.time()
    for _ in range(n):
        mx_data = step(mx_data)
        traj.append(np.array(mx_data.qpos))
    jax.block_until_ready(mx_data.qpos)
    wall = time.time() - t0
    traj = np.array(traj)

    # native reference on the same model
    d = mujoco.MjData(mj_model)
    kid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "hang")
    mujoco.mj_resetDataKeyframe(mj_model, d, kid)
    ref = [np.array(d.qpos)]
    for _ in range(n):
        mujoco.mj_step(mj_model, d)
        ref.append(np.array(d.qpos))
    ref = np.array(ref)

    z_mjx, z_ref = traj[:, 2], ref[:, 2]
    print(f"[4] {n} steps ({n*mj_model.opt.timestep:.2f} s sim) in {wall:.1f}s "
          f"-> {n/wall:.1f} steps/s ({args.impl})")
    print(f"    pelvis z: MJX start {z_mjx[0]:.4f} end {z_mjx[-1]:.4f}  "
          f"drift {z_mjx[-1]-z_mjx[0]:+.4f} m")
    print(f"    native  : start {z_ref[0]:.4f} end {z_ref[-1]:.4f}  "
          f"drift {z_ref[-1]-z_ref[0]:+.4f} m")
    print(f"    max |qpos diff| (MJX vs native) = {np.abs(traj-ref).max():.4f}")
    fell = z_mjx[-1] < z_mjx[0] - 0.30
    print(f"[5] verdict: {'FELL' if fell else 'HANG HELD'}   "
          f"({'usable' if not fell else 'not usable'})")
    return 0 if not fell else 1


if __name__ == "__main__":
    raise SystemExit(main())
