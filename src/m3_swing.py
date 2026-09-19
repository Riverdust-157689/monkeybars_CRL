"""M3.0 feasibility: can a scripted controller swing the free hand to the NEXT bar?

No learning here.  One hand keeps its grip on bar 0; a periodic controller drives a
chosen joint group; we measure how close the free (right) hand gets to bar 1 and
whether it ever *physically* touches it (via real MuJoCo contacts, not the
geometric proxy).

This runs native MuJoCo on the SAME exported scene the RL env trains on
(``assets/g1_brachiation/scene_bars.xml``), so it needs no GPU/JAX and answers the
first question in ``docs/M2_M3诊断.md``: is the block exploration, or is the
geometry/torque/controller simply unable to reach the next bar?

Usage::

    .venv-warp/bin/python src/m3_swing.py                     # sweep every mechanism
    .venv-warp/bin/python src/m3_swing.py --mode free_arm --freq 0.5 --amp 0.4
    .venv-warp/bin/python src/m3_swing.py --mode all --top 15
    .venv-warp/bin/python src/m3_swing.py --mode free_arm --grip trigger

Reported per trial:
  ``min_d_RB1``   min over time of |p_R - bar1 axis|   (p_R = the env's grasp centre)
  ``B1 hit``      1 if the right hand ever CONTACTED bar1's geoms
  ``swing``       max |torso x - x0|  (pendulum amplitude)
  ``|v|max``      max |torso linear velocity|
  ``Lgrip%``      fraction of the trial with left-hand ↔ bar contact
  ``fell``        torso dropped below the fall depth
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "envs"), os.path.join(REPO, "assets", "g1_brachiation")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco  # noqa: E402

import build_scene as bs  # noqa: E402
from envs.brachiation import ACTION_WINDOWS, ARM_WAIST_ACT, FINGER_JOINTS  # noqa: E402  (single source of truth)

SCENE = os.path.join(REPO, "assets", "g1_brachiation", "scene_bars.xml")
# Mirrors `create_brachiation`'s registry, kept local so this tool stays pure
# MuJoCo/CPU (importing the env pulls in jax).  "full035" is the M3.1
# easy-geometry asset: same build parameters as "full", only the bar spacing is
# 0.35 instead of 0.40 (the hang keyframe is solved from B0, which does not move,
# so the keyframe and B0 are bit-identical between the two).
SCENES = {
    "full": SCENE,
    "full035": os.path.join(REPO, "assets", "g1_brachiation", "scene_bars_d035.xml"),
}
TAU_CONTACT = 0.04
FALL_DEPTH = 0.8
GRASP_THRESH = 0.05

MODES = ("free_arm", "free_full", "waist", "elbow", "shoulder", "all", "reach_ik")


def joint_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def act_of(m, joint_name):
    return int(np.flatnonzero(m.actuator_trnid[:, 0] == joint_id(m, joint_name))[0])


class Rig:
    """Scene + index bookkeeping + the env's geometric grasp points."""

    def __init__(self, scene: str = SCENE, spacing: float | None = None,
                 mjx_compat: bool = False):
        if spacing is not None:
            # Build in memory with a different bar spacing (the builder re-solves the
            # hang keyframe for that geometry) so the easy-geometry fallback is testable.
            # mjx_compat=False matches the asset the RL env trains on (--scene full:
            # full mesh, cage_offset (0.015,-0.036)); mjx_compat=True is the capsule
            # variant, which has a DIFFERENT keyframe and reach (see scene_bars*.json).
            self.m, self.solved = bs.build_model(
                bs.SceneParams(spacing=float(spacing), mjx_compat=bool(mjx_compat)),
                verbose=False)
        else:
            self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "hang")
        self.key_ctrl = m.key_ctrl[self.kid].copy()
        self.key_qpos = m.key_qpos[self.kid].copy()
        self.act = {n: act_of(m, n) for n in ARM_WAIST_ACT}
        self.fing = {s: [act_of(m, f"{s}_hand_{j}_joint") for j in FINGER_JOINTS]
                     for s in ("left", "right")}
        self.wrist = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_roll_link")
                      for s in ("left", "right")}
        self.bar_geoms = bs.bar_geom_ids(m)
        self.bar_x = np.array([m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"bar{i}")][0]
                               for i in range(5)])
        b0 = np.array([self.bar_x[0], 0.0, m.body_pos[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bar0")][2]])
        self.bar_z = float(b0[2])
        # seat_local: bar0 expressed in each wrist frame at the keyframe (same as the env)
        self.seat = {}
        md = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, md, self.kid)
        mujoco.mj_forward(m, md)
        for s, bid in self.wrist.items():
            R = md.xmat[bid].reshape(3, 3)
            self.seat[s] = R.T @ (b0 - md.xpos[bid])

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.kid)
        self.d.ctrl[:] = self.key_ctrl
        mujoco.mj_forward(self.m, self.d)

    def hand_point(self, side: str) -> np.ndarray:
        d = self.d
        bid = self.wrist[side]
        return d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ self.seat[side]

    def dist_to_bar(self, side: str, k: int) -> float:
        p = self.hand_point(side)
        return float(np.hypot(p[0] - self.bar_x[k], p[2] - self.bar_z))

    def contacts_vs_bars(self):
        """(n_contacts_per_bar_index, n_left, n_right) from REAL MuJoCo contacts."""
        rep = bs.contact_report(self.m, self.d)
        per_bar = {i: 0 for i in range(len(self.bar_geoms))}
        for g in rep["bars_touched"]:
            if g in self.bar_geoms:
                per_bar[self.bar_geoms.index(g)] += 1
        return per_bar, rep["left"], rep["right"]

    def set_grip(self, side: str, c: float):
        base = self.key_ctrl[self.fing[side]]
        self.d.ctrl[self.fing[side]] = c * base


def ik_reach(rig: Rig, pitch_rng, roll_rng, elbow_rng):
    """Static kinematic sweep: move ONLY the free (right) arm, keep everything else
    at the hang keyframe, and find the closest the grasp centre can get to B1.

    Pure ``mj_forward`` (no integration), so this separates "the geometry allows it"
    from "the controller/exploration must find it".
    """
    m, d = rig.m, rig.d
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    qadr = {n: m.jnt_qposadr[jid(n)] for n in
            ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint")}
    base = rig.key_qpos.copy()
    best = dict(d=np.inf)
    for sp in pitch_rng:
        for sr in roll_rng:
            for el in elbow_rng:
                q = base.copy()
                q[qadr["right_shoulder_pitch_joint"]] = sp
                q[qadr["right_shoulder_roll_joint"]] = sr
                q[qadr["right_elbow_joint"]] = el
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                dist = rig.dist_to_bar("right", 1)
                if dist < best["d"]:
                    best = dict(d=dist, sp=float(sp), sr=float(sr), el=float(el),
                                p=rig.hand_point("right").copy(),
                                dL0=rig.dist_to_bar("left", 0))
    return best


def action_box_reach(rig: Rig, n: int = 20000, seed: int = 0, free_arm_only: bool = False,
                     window: str = "m1"):
    """Best |p_R - B1| reachable by the POLICY'S ACTION PARAMETERIZATION.

    The env commands ``q_des = key_ctrl + ARM_WAIST_SCALE * a`` with ``a in [-1, 1]``,
    so the reachable joint window is only +-SCALE around the hang keyframe.  This
    random-searches that box kinematically (body frozen, left hand exactly on B0) and
    reports how close the free hand can get -- i.e. whether the action space can even
    *express* a reaching pose.
    """
    m, d = rig.m, rig.d
    scale = ACTION_WINDOWS[window]
    act_names = list(ARM_WAIST_ACT)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in act_names]
    qadr = [m.jnt_qposadr[j] for j in jids]
    rng = np.random.default_rng(seed)
    base = rig.key_qpos.copy()
    best = dict(d=np.inf)
    for i in range(n):
        a = rng.uniform(-1.0, 1.0, len(act_names))
        if free_arm_only:                      # only the right arm (5 dims) may move
            a[:5] = 0.0
        q = base.copy()
        for k, adr in enumerate(qadr):
            q = q.at[adr] if hasattr(q, "at") else q
            q[adr] = base[adr] + scale[k] * a[k]
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        dist = rig.dist_to_bar("right", 1)
        if dist < best["d"]:
            best = dict(d=dist, a=a.copy(), p=rig.hand_point("right").copy(),
                        dL0=rig.dist_to_bar("left", 0))
    return best


def controller(rig: Rig, mode: str, f: float, amp: float, phase: float, t: float,
               grip_trigger: bool, catch_r: float, bias: float = 0.0,
               roll_cmd: float = 0.0, elbow_cmd: float = 0.0) -> float:
    """Write ctrl for time t; returns the (soft) right-hand closure command."""
    s = np.sin(2 * np.pi * f * t)
    s_ph = np.sin(2 * np.pi * f * t + phase)
    kc = rig.key_ctrl
    A = rig.act

    def add(name, delta):
        rig.d.ctrl[A[name]] = kc[A[name]] + delta

    if mode == "reach_ik":
        # static target = the best kinematic pose (see ik_reach); a slight oscillation
        # on top helps the fingers slide off bar 0 in the first tenths of a second
        add("right_shoulder_pitch_joint", bias + amp * s)
        add("right_shoulder_roll_joint", roll_cmd)
        add("right_elbow_joint", elbow_cmd)
    if mode in ("free_arm", "all"):
        add("right_shoulder_pitch_joint", bias + amp * s_ph)
    if mode == "free_full":
        add("right_shoulder_pitch_joint", bias + amp * s_ph)
        add("right_shoulder_roll_joint", -0.4 * abs(bias) - 0.2 * amp * (1 - np.cos(2*np.pi*f*t + phase)) / 2)
        # extend the elbow as the arm comes forward (reach), fold it on the backswing
        add("right_elbow_joint", -0.6 * amp * (1.0 - np.cos(2 * np.pi * f * t + phase)) / 2)
    if mode in ("waist", "all"):
        add("waist_pitch_joint", amp * s_ph)
    if mode in ("elbow", "all"):
        add("left_elbow_joint", 0.5 * amp * s_ph)          # pendulum length pumping
    if mode == "shoulder":
        add("left_shoulder_pitch_joint", 0.35 * amp * s_ph)

    # grippers: both stay closed unless we are testing a "release and catch"
    c_r = 1.0
    if grip_trigger:
        c_r = 1.0 if rig.dist_to_bar("right", 1) < catch_r else 0.0
    rig.set_grip("left", 1.0)
    rig.set_grip("right", c_r)
    return c_r


def trial(rig: Rig, mode: str, f: float, amp: float, phase: float, seconds: float,
          grip_trigger: bool, catch_r: float, record: bool = False, contact_every: int = 10,
          bias: float = 0.0, roll_cmd: float = 0.0, elbow_cmd: float = 0.0):
    rig.reset()
    m, d = rig.m, rig.d
    n = int(seconds / m.opt.timestep)
    x0 = float(d.qpos[0])
    out = dict(min_d_RB1=np.inf, min_d_LB0=np.inf, swing=0.0, vmax=0.0, wmax=0.0,
               Lgrip=0, B1hit=0, fell=0, frames=[])
    for i in range(n):
        t = i * m.opt.timestep
        controller(rig, mode, f, amp, phase, t, grip_trigger, catch_r, bias,
                   roll_cmd, elbow_cmd)
        mujoco.mj_step(m, d)
        pR = rig.hand_point("right")
        dl1 = float(np.hypot(pR[0] - rig.bar_x[1], pR[2] - rig.bar_z))
        out["min_d_RB1"] = min(out["min_d_RB1"], dl1)
        out["min_d_LB0"] = min(out["min_d_LB0"], rig.dist_to_bar("left", 0))
        out["swing"] = max(out["swing"], abs(float(d.qpos[0]) - x0))
        out["vmax"] = max(out["vmax"], float(np.linalg.norm(d.qvel[:3])))
        out["wmax"] = max(out["wmax"], float(np.linalg.norm(d.qvel[3:6])))
        if i % contact_every == 0:
            per_bar, nl, nr = rig.contacts_vs_bars()
            out["Lgrip_steps"] = out.get("Lgrip_steps", 0) + int(nl > 0)
            out["Lgrip_total"] = out.get("Lgrip_total", 0) + 1
            if per_bar.get(1, 0) > 0:
                out["B1hit"] = 1
        if d.qpos[2] < rig.bar_z - FALL_DEPTH:
            out["fell"] = 1
        if record and i % max(1, int(0.02 / m.opt.timestep)) == 0:
            out["frames"].append((d.qpos.copy(), rig.hand_point("right")[0]))
    out["Lgrip"] = out.get("Lgrip_steps", 0) / max(1, out.get("Lgrip_total", 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=list(MODES))
    ap.add_argument("--freq", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--amp", type=float, nargs="+", default=[0.10, 0.25, 0.45])
    ap.add_argument("--phase", type=float, nargs="+",
                    default=[0.0, 1.5708, 3.1416, 4.7124])
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="trial length; a resonant pump needs several periods "
                         "(pendulum T ~ 1.7 s for this arm length)")
    ap.add_argument("--grip", default="closed", choices=["closed", "trigger"],
                    help="closed: right hand stays closed; trigger: open, close within --catch-r")
    ap.add_argument("--catch-r", type=float, default=0.07)
    ap.add_argument("--bias", type=float, default=0.0,
                    help="constant offset on the driven free-arm shoulder pitch (rad); "
                         "negative rotates the arm forward/down from the hang pose")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--video", default=None, help="render the best trial to this mp4")
    ap.add_argument("--ik", action="store_true",
                    help="only run the static kinematic reach sweep on the free arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default="full", choices=sorted(SCENES),
                    help="which exported asset to load (full = 0.40 m bars, the asset all "
                         "runs so far used; full035 = 0.35 m bars, M3.1 easy geometry). "
                         "Ignored when --spacing is given (that builds in memory instead)")
    ap.add_argument("--mjx-compat", type=int, default=0,
                    help="with --spacing: build the capsule/mjx variant instead of the "
                         "full-mesh asset the RL env uses (they have different keyframes)")
    ap.add_argument("--spacing", type=float, default=None,
                    help="build the scene in memory with this bar spacing (default 0.40 from "
                         "the exported XML); tests the M3 easy-geometry fallback")
    ap.add_argument("--goto-ik", action="store_true",
                    help="drive the free arm to the IK-optimal pose THROUGH the action "
                         "window (q_des = key + scale*a) and measure the dynamic reach")
    ap.add_argument("--refine-action", type=int, default=0,
                    help="random-search N samples of the POLICY action box WITH dynamics "
                         "(right arm + waist; left arm held at the keyframe so the support "
                         "grip is maintained by the physics)")
    ap.add_argument("--window", default="m1", choices=sorted(ACTION_WINDOWS),
                    help="action window to use for --refine-action")
    ap.add_argument("--action-box", type=int, default=0,
                    help="random-search N samples of the POLICY action box "
                         "(q_des = key + scale*a, a in [-1,1]) and report the best "
                         "|p_R-B1| the action space can even express")
    ap.add_argument("--refine", type=int, default=0,
                    help="random-search N free-arm poses around the IK optimum WITH dynamics "
                         "(a 'dynamic IK': can a hand-designed pose actually grasp B1 while "
                         "the other hand holds B0?)")
    args = ap.parse_args()

    modes = MODES if args.mode == "all" else (args.mode,)
    rig = Rig(scene=SCENES[args.scene], spacing=args.spacing,
              mjx_compat=bool(args.mjx_compat))
    ik = None
    src = ("in-memory build (--spacing)" if args.spacing is not None
           else f"{os.path.basename(SCENES[args.scene])} (exported XML)")
    print(f"[scene] {src}  spacing={rig.bar_x[1]-rig.bar_x[0]:.4f} m  "
          f"bar_x={np.round(rig.bar_x, 4).tolist()} bar_z={rig.bar_z:.4f}")
    rig.reset()
    base_gap = rig.dist_to_bar("right", 1)
    print(f"[baseline] static hang: |p_R - B1| = {base_gap:.4f} m "
          f"(grasp threshold {GRASP_THRESH} m); |p_L - B0| = {rig.dist_to_bar('left', 0):.4f} m")

    if args.ik:
        best = ik_reach(rig,
                        np.linspace(-3.0, 1.0, 41),      # shoulder pitch
                        np.linspace(-2.0, 2.0, 21),      # shoulder roll
                        np.linspace(0.0, 2.5, 21))       # elbow
        p = best["p"]
        print(f"[ik] best static right-arm pose: shoulder_pitch={best['sp']:+.3f} "
              f"roll={best['sr']:+.3f} elbow={best['el']:.3f}")
        print(f"[ik] p_R = ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})  ->  "
              f"|p_R - B1| = {best['d']:.4f} m   (B1 at x={rig.bar_x[1]:.4f}, z={rig.bar_z:.4f})")
        print(f"[ik] left hand still at |p_L - B0| = {best['dL0']:.4f} m")
        if best["d"] < GRASP_THRESH:
            print("[verdict] a STATIC pose reaches B1 -> geometry is fine; "
                  "the RL block is exploration/control, not reachability")
        elif best["d"] < 0.15:
            print("[verdict] static reach gets within 0.15 m -> reachable with a little "
                  "body swing; geometry is fine")
        else:
            print("[verdict] even the best static arm pose cannot approach B1 -> the left "
                  "hand cannot stay on B0 while the right reaches B1; the first crossing "
                  "REQUIRES releasing/rotating the body (or a bigger gap reduction)")
        return 0

    ik = ik_reach(rig, np.linspace(-3.0, 1.0, 41), np.linspace(-2.0, 2.0, 21),
                  np.linspace(0.0, 2.5, 21))
    print(f"[ik] best static free-arm pose: sp={ik['sp']:+.3f} sr={ik['sr']:+.3f} "
          f"el={ik['el']:.3f} -> |p_R-B1| = {ik['d']:.4f} m")

    if args.goto_ik:
        scale = np.array(ACTION_WINDOWS[args.window])
        kc = rig.key_ctrl
        want = {"right_shoulder_pitch_joint": ik["sp"], "right_shoulder_roll_joint": ik["sr"],
                "right_elbow_joint": ik["el"]}
        a = np.zeros(len(ARM_WAIST_ACT))
        for k, n in enumerate(ARM_WAIST_ACT):
            if n in want:
                a[k] = (want[n] - kc[rig.act[n]]) / scale[k]
        a_clipped = np.clip(a, -1.0, 1.0)
        rig.reset()
        n = int(args.seconds / rig.m.opt.timestep)
        min_d, hit, lg, lgn, fell = np.inf, 0, 0, 0, 0
        for t in range(n):
            for k, nm in enumerate(ARM_WAIST_ACT):
                rig.d.ctrl[rig.act[nm]] = np.clip(
                    kc[rig.act[nm]] + scale[k] * a_clipped[k],
                    rig.m.actuator_ctrlrange[rig.act[nm]][0],
                    rig.m.actuator_ctrlrange[rig.act[nm]][1])
            rig.set_grip("left", 1.0)
            rig.set_grip("right", 1.0 if rig.dist_to_bar("right", 1) < args.catch_r else 0.0)
            mujoco.mj_step(rig.m, rig.d)
            min_d = min(min_d, rig.dist_to_bar("right", 1))
            if t % 10 == 0:
                per_bar, nl, _ = rig.contacts_vs_bars()
                lgn += 1
                lg += int(nl > 0)
                hit = hit or int(per_bar.get(1, 0) > 0)
            fell = fell or int(rig.d.qpos[2] < rig.bar_z - FALL_DEPTH)
        print(f"[goto-ik] spacing={rig.bar_x[1]-rig.bar_x[0]:.2f} window={args.window}: "
              f"required a={np.round(a,2).tolist()} -> clipped a={np.round(a_clipped,2).tolist()}")
        print(f"[goto-ik] dynamic min|p_R-B1| = {min_d:.4f} m, B1 contact = {hit}, "
              f"left grip kept {lg/max(1,lgn)*100:.0f}%, fell = {fell}")
        return 0

    if args.refine_action:
        scale = np.array(ACTION_WINDOWS[args.window])
        rng = np.random.default_rng(args.seed)
        kc = rig.key_ctrl
        kq = rig.key_qpos
        rows = []
        for i in range(args.refine_action):
            a = rng.uniform(-1.0, 1.0, len(ARM_WAIST_ACT))
            a[:5] = 0.0                       # left arm stays at the keyframe (support)
            rig.reset()

            def drive(rig, a=a):
                for k, n in enumerate(ARM_WAIST_ACT):
                    rig.d.ctrl[rig.act[n]] = np.clip(
                        kc[rig.act[n]] + scale[k] * a[k],
                        rig.m.actuator_ctrlrange[rig.act[n]][0],
                        rig.m.actuator_ctrlrange[rig.act[n]][1])
                rig.set_grip("left", 1.0)
                c = 1.0 if rig.dist_to_bar("right", 1) < args.catch_r else 0.0
                rig.set_grip("right", c)

            n = int(args.seconds / rig.m.opt.timestep)
            min_d, hit, lgrip, lgrip_n, fell = np.inf, 0, 0, 0, 0
            for t in range(n):
                drive(rig)
                mujoco.mj_step(rig.m, rig.d)
                min_d = min(min_d, rig.dist_to_bar("right", 1))
                if t % 10 == 0:
                    per_bar, nl, _ = rig.contacts_vs_bars()
                    lgrip_n += 1
                    lgrip += int(nl > 0)
                    hit = hit or int(per_bar.get(1, 0) > 0)
                fell = fell or int(rig.d.qpos[2] < rig.bar_z - FALL_DEPTH)
            rows.append(dict(a=a.copy(), d=min_d, hit=hit, Lgrip=lgrip / max(1, lgrip_n), fell=fell))
        rows.sort(key=lambda r: r["d"])
        print(f"\n[refine-action] window={args.window}  {len(rows)} samples of a in [-1,1]^12 "
              f"(right arm + waist), {args.seconds}s each")
        for r in rows[:args.top]:
            print(f"  |p_R-B1|={r['d']:.4f}  B1hit={r['hit']}  Lgrip={r['Lgrip']*100:.0f}%  "
                  f"fell={r['fell']}  a={np.round(r['a'],2).tolist()}")
        n_ok = sum(1 for r in rows if r["d"] < GRASP_THRESH)
        n_hit = sum(r["hit"] for r in rows)
        print(f"[verdict] window={args.window}: {n_ok}/{len(rows)} samples get the grasp centre "
              f"within {GRASP_THRESH} m, {n_hit}/{len(rows)} physically CONTACT bar 1")
        return 0

    if args.action_box:
        for label, kw in (("all 12 arm+waist dims", dict(free_arm_only=False)),
                          ("right arm only (5 dims)", dict(free_arm_only=True))):
            for win in ("m1", "reach"):
                b = action_box_reach(rig, args.action_box, args.seed, window=win, **kw)
                print(f"[action-box] window={win:5s} {label:24s} best |p_R-B1| = {b['d']:.4f} m "
                      f"(left hand at B0: {b['dL0']:.4f})")
            print(f"[action-box] {label:26s} best |p_R-B1| = {b['d']:.4f} m  "
                  f"(left hand at B0: {b['dL0']:.4f})  a={np.round(b['a'],2).tolist()}")
        print(f"[note] grasp threshold {GRASP_THRESH} m; static-kinematics optimum with a FREE "
              f"arm is 0.0310 m at d=0.40 -- the gap to that number is the cost of the "
              f"+-(ARM_WAIST_SCALE*1) action window")
        return 0

    if args.refine:
        rng = np.random.default_rng(args.seed)
        kc = rig.key_ctrl
        off = lambda n, v: v - kc[rig.act[n]]
        rows = []
        for i in range(args.refine):
            sp = ik["sp"] + rng.normal(0, 0.30)
            sr = ik["sr"] + rng.normal(0, 0.30)
            el = ik["el"] + rng.normal(0, 0.30)
            r = trial(rig, "reach_ik", 0.0, 0.0, 0.0, args.seconds, True, args.catch_r,
                      bias=off("right_shoulder_pitch_joint", sp),
                      roll_cmd=off("right_shoulder_roll_joint", sr),
                      elbow_cmd=off("right_elbow_joint", el))
            rows.append(dict(sp=sp, sr=sr, el=el, **r))
        rows.sort(key=lambda r: r["min_d_RB1"])
        print(f"\n[dynamic-IK] {len(rows)} hand-designed poses, {args.seconds}s each")
        print(f"{'sp':>7} {'sr':>7} {'el':>7} {'min_d_RB1':>10} {'B1hit':>6} {'Lgrip%':>7} {'fell':>5}")
        for r in rows[:args.top]:
            print(f"{r['sp']:>7.3f} {r['sr']:>7.3f} {r['el']:>7.3f} {r['min_d_RB1']:>10.4f} "
                  f"{r['B1hit']:>6d} {r['Lgrip']*100:>6.1f}% {r['fell']:>5d}")
        hit = sum(r["B1hit"] for r in rows)
        print(f"\n[verdict] {hit}/{len(rows)} hand-designed poses physically CONTACT bar 1; "
              f"best |p_R-B1| = {rows[0]['min_d_RB1']:.4f} m (threshold {GRASP_THRESH})")
        if hit:
            b = rows[0]
            print("[result] scripted existence proof of the first crossing: left hand holds B0, "
                  "right hand reaches B1 -> the RL block is exploration, not reachability")
        return 0

    rows = []
    t0 = time.time()
    for mode in modes:
        for f in args.freq:
            for amp in args.amp:
                for ph in args.phase:
                    # reach_ik drives the free arm to an ABSOLUTE target pose, so the
                    # offsets are measured against the keyframe ctrl
                    if mode == "reach_ik":
                        kc = rig.key_ctrl
                        b = ik["sp"] - kc[rig.act["right_shoulder_pitch_joint"]]
                        rc = ik["sr"] - kc[rig.act["right_shoulder_roll_joint"]]
                        ec = ik["el"] - kc[rig.act["right_elbow_joint"]]
                    else:
                        b, rc, ec = args.bias, 0.0, 0.0
                    r = trial(rig, mode, f, amp, ph, args.seconds,
                              args.grip == "trigger" or mode == "reach_ik",
                              args.catch_r, bias=b, roll_cmd=rc, elbow_cmd=ec)
                    rows.append(dict(mode=mode, f=f, amp=amp, phase=ph, bias=args.bias, **{
                        k: v for k, v in r.items() if k != "frames"}))
    rows.sort(key=lambda r: (r["min_d_RB1"]))
    print(f"\n[{len(rows)} trials, {time.time()-t0:.1f}s] best by min|p_R - B1|"
          f"   ({args.grip} grip, {args.seconds}s each)")
    print(f"{'mode':10s} {'f':>5} {'amp':>5} {'bias':>6} {'phase':>6} {'min_d_RB1':>10} "
          f"{'B1hit':>6} {'swing':>7} {'|v|max':>7} {'Lgrip%':>7} {'fell':>5}")
    for r in rows[:args.top]:
        print(f"{r['mode']:10s} {r['f']:>5.2f} {r['amp']:>5.2f} {r['bias']:>6.2f} {r['phase']:>6.2f} "
              f"{r['min_d_RB1']:>10.4f} {r['B1hit']:>6d} {r['swing']:>7.3f} "
              f"{r['vmax']:>7.3f} {r['Lgrip']*100:>6.1f}% {r['fell']:>5d}")

    best = rows[0]
    print(f"\n[best] {best['mode']} f={best['f']} amp={best['amp']} phase={best['phase']}: "
          f"min|p_R-B1| = {best['min_d_RB1']:.4f} m "
          f"({'CONTACT with B1!' if best['B1hit'] else 'no B1 contact'}), "
          f"swing {best['swing']:.3f} m, left grip kept {best['Lgrip']*100:.0f}% of the time")
    if best["B1hit"]:
        print("[verdict] the mechanism CAN touch the next bar -> the RL block is exploration, "
              "not geometry/torque")
    elif best["min_d_RB1"] < 0.2:
        print("[verdict] gets close (<0.2 m) but no contact -> reachable region exists; "
              "exploration + goal shaping should be enough")
    else:
        print("[verdict] NOT reachable with these scripted patterns -> check geometry/torque/"
              "controller BEFORE blaming CRL (see docs/M2_M3诊断.md item 1)")

    if args.video:
        try:
            import imageio.v2 as imageio
            r = trial(rig, best["mode"], best["f"], best["amp"], best["phase"],
                      args.seconds, args.grip == "trigger", args.catch_r, record=True,
                      bias=best.get("bias", 0.0))
            renderer = mujoco.Renderer(rig.m, 480, 640)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.azimuth, cam.elevation, cam.distance = 70.0, -8.0, 3.4
            cam.lookat[:] = [0.8, 0.0, rig.bar_z - 0.45]
            frames = []
            for qpos, _ in r["frames"]:
                rig.d.qpos[:] = qpos
                mujoco.mj_forward(rig.m, rig.d)
                renderer.update_scene(rig.d, cam)
                frames.append(renderer.render())
            os.makedirs(os.path.dirname(args.video), exist_ok=True)
            imageio.mimsave(args.video, frames, fps=50, macro_block_size=1)
            print(f"[video] {args.video} ({len(frames)} frames)")
        except Exception as exc:
            print(f"[warn] video failed ({type(exc).__name__}: {exc}); "
                  f"needs MUJOCO_GL=egl or a display")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
