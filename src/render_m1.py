"""Render M1 evidence videos and run a long-horizon stability check.

Usage::

    MUJOCO_GL=glfw .venv/bin/python src/render_m1.py [--long 60]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "assets", "g1_brachiation"))

import build_scene as bs  # noqa: E402

OUT = os.path.join(REPO, "runs", "m1")


def camera_for(model, data, azim=70, elev=-8, dist=2.4):
    cam = mujoco.MjvCamera()
    b0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bar0")
    cam.lookat[:] = [model.body_pos[b0][0], 0.0, model.body_pos[b0][2] - 0.45]
    cam.distance = dist
    cam.azimuth = azim
    cam.elevation = elev
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    return cam


def record(model, data, out_path, n_steps, target_fn=None, fps=30, size=(480, 640), camera=None):
    renderer = mujoco.Renderer(model, *size)
    cam = camera or camera_for(model, data)
    dt = model.opt.timestep
    every = max(1, int(round(1.0 / (fps * dt))))
    frames = []
    for step in range(n_steps):
        if target_fn is not None:
            target_fn(step * dt, data)
        mujoco.mj_step(model, data)
        if step % every == 0:
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
    imageio.mimsave(out_path, frames, fps=fps)
    return len(frames)


def leg_swing_target(model, amp=0.55, freq=0.6, settle=0.5):
    """Return a ``fn(t, data)`` that pumps the legs while the hands hold."""
    ctrl0 = model.key_ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hang")]
    hip = [int(np.flatnonzero(model.actuator_trnid[:, 0] == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_hip_pitch_joint"))[0]) for s in ("left", "right")]
    knee = [int(np.flatnonzero(model.actuator_trnid[:, 0] == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_knee_joint"))[0]) for s in ("left", "right")]
    base_hip, base_knee = ctrl0[hip[0]], ctrl0[knee[0]]

    def fn(t, d):
        s = 0.0 if t < settle else np.sin(2 * np.pi * freq * (t - settle))
        d.ctrl[:] = ctrl0
        for i in hip:
            d.ctrl[i] = base_hip + amp * s
        for i in knee:
            d.ctrl[i] = base_knee - amp * s
    return fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", type=float, default=60.0, help="long-horizon hang seconds")
    ap.add_argument("--elbow", type=float, default=None)
    args = ap.parse_args()

    kwargs = {} if args.elbow is None else {"elbow": args.elbow}
    model, solved = bs.build_model(bs.SceneParams(), verbose=True, **kwargs)
    data = mujoco.MjData(model)
    os.makedirs(OUT, exist_ok=True)
    print(f"[scene] mass={solved['info']['total_mass']:.2f} kg bar={np.round(solved['bar_xy'],4)} "
          f"dx={solved['dx']:+.4f} m")

    # ---- long-horizon stability -------------------------------------------
    bs.reset_to(model, data)
    data.ctrl[:] = model.key_ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hang")]
    z0 = float(data.qpos[2])
    zs, ls, rs = [], [], []
    for step in range(int(args.long / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if step % int(0.5 / model.opt.timestep) == 0:
            rep = bs.contact_report(model, data)
            zs.append(float(data.qpos[2]))
            ls.append(rep["left"])
            rs.append(rep["right"])
    zs = np.array(zs)
    print(f"[long hang {args.long:.0f}s] pelvis z: start {z0:.4f} end {zs[-1]:.4f} "
          f"drift {zs[-1]-z0:+.4f} m, min {zs.min():.4f}, max {zs.max():.4f}")
    print(f"[long hang] grasp present: left {np.mean(np.array(ls)>0):.3f} "
          f"right {np.mean(np.array(rs)>0):.3f}, min contacts L={min(ls)} R={min(rs)}")

    # ---- videos ------------------------------------------------------------
    for name, n, fn in (
        ("hang_static", 6.0, None),
        ("leg_swing", 12.0, leg_swing_target(model)),
    ):
        bs.reset_to(model, data)
        data.ctrl[:] = model.key_ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hang")]
        path = os.path.join(OUT, f"{name}.mp4")
        frames = record(model, data, path, int(n / model.opt.timestep), fn)
        print(f"[video] {path} ({frames} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
