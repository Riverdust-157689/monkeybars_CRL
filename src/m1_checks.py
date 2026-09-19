"""M1 acceptance checks for the G1 monkey-bar scene (scripted, no learning).

M1 answers one question: **is the hanging task physically feasible in MuJoCo at
all?**  Four scripted tests, all starting from the settled ``hang`` keyframe:

  M1.1  static two-hand hang            -- hold 20 s, both grasps persist
  M1.2  single-hand hang                -- release one hand, hold through a swing
  M1.3  release -> re-grasp on same bar -- the contact-switching primitive
  M1.4  lower-body swing while hanging  -- pump the legs, grasp survives

Every test returns metrics plus a PASS/FAIL verdict and the reason.

Run::

    .venv/bin/python src/m1_checks.py                 # all checks
    .venv/bin/python src/m1_checks.py --check M1.2    # one check
    .venv/bin/python src/m1_checks.py --grip-sweep    # M1.2 vs grip torque scale
    .venv/bin/python src/m1_checks.py --markdown runs/m1/report.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Dict, List

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "assets", "g1_brachiation"))

import build_scene as bs  # noqa: E402

OUT_DIR = os.path.join(REPO, "runs", "m1")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def actuator_index(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    hits = np.flatnonzero(model.actuator_trnid[:, 0] == jid)
    if len(hits) == 0:
        raise KeyError(joint_name)
    return int(hits[0])


def hand_actuators(model: mujoco.MjModel, side: str) -> List[int]:
    return [i for i in range(model.nu)
            if f"{side}_hand_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")]


def leg_actuators(model: mujoco.MjModel, joint: str) -> List[int]:
    return [actuator_index(model, f"{s}_{joint}_joint") for s in ("left", "right")]


def torque_utilisation(model: mujoco.MjModel, data: mujoco.MjData) -> Dict[str, float]:
    """max |qfrc_actuator| / actfrcrange, per joint group."""
    groups: Dict[str, List[float]] = {"arm": [], "hand": [], "leg": [], "waist": []}
    worst = ("", 0.0)
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name or not model.jnt_actfrclimited[jid]:
            continue
        limit = max(abs(model.jnt_actfrcrange[jid]))
        if limit <= 0:
            continue
        util = abs(data.qfrc_actuator[model.jnt_dofadr[jid]]) / limit
        group = ("hand" if "_hand_" in name else
                 "waist" if name.startswith("waist") else
                 "leg" if any(k in name for k in ("hip", "knee", "ankle")) else "arm")
        groups[group].append(float(util))
        if util > worst[1]:
            worst = (name, float(util))
    out = {g: (max(v) if v else 0.0) for g, v in groups.items()}
    out["worst_joint"] = worst[0]
    out["worst_util"] = worst[1]
    return out


def run(model: mujoco.MjModel, data: mujoco.MjData, n_steps: int,
        target_fn=None, sample: float = 0.05) -> Dict[str, np.ndarray]:
    """Step the sim and record a trace. ``target_fn(t, data)`` may edit ctrl."""
    dt = model.opt.timestep
    k = max(1, int(round(sample / dt)))
    keys = ("t", "pelvis", "com", "ncon_bar", "left", "right", "min_dist", "qvel", "foot")
    trace: Dict[str, list] = {key: [] for key in keys}
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    for step in range(n_steps):
        if target_fn is not None:
            target_fn(step * dt, data)
        mujoco.mj_step(model, data)
        if step % k == 0:
            rep = bs.contact_report(model, data)
            trace["t"].append((step + 1) * dt)
            trace["pelvis"].append(data.qpos[:3].copy())
            trace["com"].append(data.subtree_com[0].copy())
            trace["ncon_bar"].append(rep["n_bar_contacts"])
            trace["left"].append(rep["left"])
            trace["right"].append(rep["right"])
            trace["min_dist"].append(rep["min_dist"])
            trace["qvel"].append(float(np.linalg.norm(data.qvel)))
            trace["foot"].append(data.xpos[foot_id].copy())
    return {k: np.array(v) for k, v in trace.items()}


def key_ctrl(model: mujoco.MjModel) -> np.ndarray:
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hang")
    return model.key_ctrl[kid].copy()


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def check_static_hang(model, data, duration=20.0, **_) -> Dict:
    bs.reset_to(model, data)
    data.ctrl[:] = key_ctrl(model)
    z0 = float(data.qpos[2])
    t0 = time.time()
    tr = run(model, data, int(duration / model.opt.timestep))
    z_dev = float(np.max(np.abs(tr["pelvis"][:, 2] - z0)))
    lfrac = float(np.mean(tr["left"] > 0))
    rfrac = float(np.mean(tr["right"] > 0))
    tor = torque_utilisation(model, data)
    verdict = (lfrac > 0.98) and (rfrac > 0.98) and (z_dev < 0.05)
    return dict(
        name="M1.1 static two-hand hang",
        passed=bool(verdict),
        duration_s=duration,
        pelvis_z_start=z0,
        pelvis_z_end=float(tr["pelvis"][-1, 2]),
        max_z_deviation_m=z_dev,
        left_grasp_fraction=lfrac,
        right_grasp_fraction=rfrac,
        final_bar_contacts=int(tr["ncon_bar"][-1]),
        final_min_dist_mm=float(tr["min_dist"][-1] * 1000),
        torque_utilisation=tor,
        wall_s=time.time() - t0,
    )


def check_single_hand(model, data, release="right", hold=8.0, **kwargs) -> Dict:
    """Release one hand; can the other one carry the body through a swing?"""
    bs.reset_to(model, data)
    ctrl0 = key_ctrl(model)
    data.ctrl[:] = ctrl0
    z0 = float(data.qpos[2])
    open_ids = hand_actuators(model, release)
    ramp = 0.3

    def target_fn(t, d):
        frac = min(1.0, t / ramp)
        for i in open_ids:
            d.ctrl[i] = (1.0 - frac) * ctrl0[i]

    tr = run(model, data, int(hold / model.opt.timestep), target_fn, sample=0.05)
    kept = tr["left"] if release == "right" else tr["right"]
    grasp_frac = float(np.mean(kept > 0))
    fell = float(tr["pelvis"][-1, 2]) < z0 - 0.30
    tor = torque_utilisation(model, data)
    verdict = (grasp_frac > 0.95) and not fell
    return dict(
        name=f"M1.2 single-hand hang ({release} released)",
        passed=bool(verdict),
        duration_s=hold,
        supporting_hand="left" if release == "right" else "right",
        grasp_present_fraction=grasp_frac,
        fell=fell,
        pelvis_z_end=float(tr["pelvis"][-1, 2]),
        com_x_swing_peak_to_peak_m=float(np.ptp(tr["com"][:, 0])),
        torque_utilisation=tor,
    )


def seat_local_offset(model: mujoco.MjModel, params: bs.SceneParams) -> np.ndarray:
    """Bar offset from the knuckle mid-point, expressed in the wrist body frame."""
    robot = bs._robot_only(params)
    solved = bs.solve_hang(params)
    data = mujoco.MjData(robot)
    bs._apply_pose(robot, data, {**solved["pose"], **bs.finger_pose(1.0)})
    bid = lambda n: mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, n)
    K = 0.5 * (data.xpos[bid("left_hand_index_0_link")] + data.xpos[bid("left_hand_middle_0_link")])
    R = data.xmat[bid("left_wrist_roll_link")].reshape(3, 3)
    return R.T @ np.array([params.cage_offset[0], 0.0, params.cage_offset[1]])


def ik_reach(model: mujoco.MjModel, data: mujoco.MjData, side: str, bar_x: float, bar_z: float,
             seat_local: np.ndarray,
             arm_joints=("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"),
             iterations: int = 140, step0: float = 0.20, regularise: float = 0.005) -> Dict[str, float]:
    """Coordinate-descent IK that puts the hand's grasp point back on the bar.

    The grasp point is ``knuckle + R_wrist @ seat_local``; the cost also rewards
    keeping the finger curl axis parallel to the bar (world Y).
    """
    qpos0 = data.qpos.copy()
    scratch = mujoco.MjData(model)
    jid = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}_joint")
           for j in arm_joints}
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    def evaluate(angles):
        scratch.qpos[:] = qpos0
        for name, value in angles.items():
            lo, hi = model.jnt_range[jid[name]]
            scratch.qpos[model.jnt_qposadr[jid[name]]] = float(np.clip(value, lo, hi))
        mujoco.mj_forward(model, scratch)
        K = 0.5 * (scratch.xpos[bid(f"{side}_hand_index_0_link")]
                   + scratch.xpos[bid(f"{side}_hand_middle_0_link")])
        R = scratch.xmat[bid(f"{side}_wrist_roll_link")].reshape(3, 3)
        grasp = K + R @ seat_local
        axial = scratch.xaxis[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_index_0_joint")][1]
        return (grasp[0] - bar_x) ** 2 + (grasp[2] - bar_z) ** 2 + 0.02 * (1.0 - abs(axial))

    angles = {j: float(data.qpos[model.jnt_qposadr[jid[j]]]) for j in arm_joints}
    q_start = dict(angles)

    def cost_of(a):
        reg = sum((a[j] - q_start[j]) ** 2 for j in arm_joints)
        return evaluate(a) + regularise * reg

    cost, step = cost_of(angles), step0
    for _ in range(iterations):
        improved = False
        for name in arm_joints:
            for delta in (step, -step):
                trial = dict(angles)
                trial[name] += delta
                c = cost_of(trial)
                if c < cost - 1e-9:
                    angles, cost, improved = trial, c, True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break
    return angles


def grasp_bar_distance(model, data, side, seat_local, bar) -> float:
    """Distance from the hand's grasp point to the bar axis (0.063 m at the keyframe)."""
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    K = 0.5 * (data.xpos[bid(f"{side}_hand_index_0_link")] + data.xpos[bid(f"{side}_hand_middle_0_link")])
    R = data.xmat[bid(f"{side}_wrist_roll_link")].reshape(3, 3)
    g = K + R @ seat_local
    return float(np.hypot(g[0] - bar[0], g[2] - bar[2]))


def hand_bar_clearance(model, data, side) -> float:
    """Smallest signed distance between the hand geoms and the bar (negative = touching)."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bar0_geom")
    return float(min(mujoco.mj_geomDistance(model, data, gid, g, 0.30, None)
                     for g in bs.hand_geom_ids(model, side)))


def check_release_regrasp(model, data, side="right", open_time=0.35, settle_time=0.5,
                          reach_time=1.0, close_time=0.5, close_dist=0.006, total=9.0, **_) -> Dict:
    """Release one hand, drive it back to the bar with IK, re-grasp and hold.

    This is the contact-switching primitive brachiation is built from.  The arm
    does not return by itself: once released, the body hangs from the other hand
    and the hand drops away, so the scripted test must *reach* back (IK) before
    closing the fingers.
    """
    bs.reset_to(model, data)
    ctrl0 = key_ctrl(model)
    data.ctrl[:] = ctrl0
    z0 = float(data.qpos[2])
    ids = hand_actuators(model, side)
    arm_ids = {j: actuator_index(model, f"{side}_{j}_joint")
               for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")}
    bar = model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bar0")].copy()
    seat = seat_local_offset(model, bs.SceneParams(grip=bs.DEFAULT.grip))

    t_open0, t_open1 = 0.2, 0.2 + open_time
    t_reach0 = t_open1 + settle_time
    t_reach1 = t_reach0 + reach_time
    t_close1 = t_reach1 + close_time
    state = {"reach": None, "close_at": None, "min_dist": 9.9, "min_clear": 9.9}

    def target_fn(t, d):
        if state["reach"] is None and t >= t_reach0:
            state["reach"] = ik_reach(model, d, side, float(bar[0]), float(bar[2]), seat)
        dist = grasp_bar_distance(model, d, side, seat, bar)
        clearance = hand_bar_clearance(model, d, side)
        state["min_dist"] = min(state["min_dist"], dist)
        state["min_clear"] = min(state["min_clear"], clearance)
        # close the fingers once the bar is physically back inside the hand
        if (state["close_at"] is None and state["reach"] is not None
                and t > t_reach0 + 0.2 and clearance < close_dist):
            state["close_at"] = t
        if t < t_open0:
            f = 0.0
        elif t < t_open1:
            f = (t - t_open0) / max(1e-6, open_time)
        elif state["close_at"] is None:
            f = 1.0
        elif t < state["close_at"] + close_time:
            f = 1.0 - (t - state["close_at"]) / max(1e-6, close_time)
        else:
            f = 0.0
        for i in ids:
            d.ctrl[i] = (1.0 - f) * ctrl0[i]
        if state["reach"] is not None and t >= t_reach0:
            alpha = min(1.0, (t - t_reach0) / max(1e-6, reach_time))
            for name, aid in arm_ids.items():
                d.ctrl[aid] = (1.0 - alpha) * ctrl0[aid] + alpha * state["reach"][name]

    tr = run(model, data, int(total / model.opt.timestep), target_fn, sample=0.05)
    seq = tr["right"] if side == "right" else tr["left"]
    t = tr["t"]
    released = bool(np.any((seq == 0) & (t > t_open1) & (t < t_reach0)))
    close_at = state["close_at"]
    after = t > (close_at + close_time + 0.3 if close_at is not None else total)
    regained = bool(np.any(seq[after] > 0)) if np.any(after) else False
    held = float(np.mean(seq[after] > 0)) if np.any(after) else 0.0
    fell = float(tr["pelvis"][-1, 2]) < z0 - 0.30
    verdict = released and regained and held > 0.8 and not fell
    return dict(
        name=f"M1.3 release -> reach -> re-grasp ({side}, same bar)",
        passed=bool(verdict),
        released=released,
        reached_with_ik=bool(state["reach"] is not None),
        regained_contact=regained,
        close_triggered_at_s=(round(float(close_at), 3) if close_at is not None else None),
        min_grasp_bar_distance_m=round(float(state["min_dist"]), 4),
        min_hand_bar_clearance_m=round(float(state["min_clear"]), 4),
        grasp_fraction_after_close=held,
        final_contacts=int(seq[-1]),
        fell=fell,
        pelvis_z_end=float(tr["pelvis"][-1, 2]),
        contact_trace=seq.tolist(),
    )


def check_leg_swing(model, data, amp=0.55, freq=0.6, duration=12.0, waist_amp=0.0, **_) -> Dict:
    """Swing the actuated legs while hanging; the grasp must survive."""
    bs.reset_to(model, data)
    ctrl0 = key_ctrl(model)
    data.ctrl[:] = ctrl0
    z0 = float(data.qpos[2])
    hip = leg_actuators(model, "hip_pitch")
    knee = leg_actuators(model, "knee")
    waist = actuator_index(model, "waist_pitch_joint")
    base_hip, base_knee, base_waist = ctrl0[hip[0]], ctrl0[knee[0]], ctrl0[waist]
    settle = 0.5

    def target_fn(t, d):
        s = 0.0 if t < settle else np.sin(2 * np.pi * freq * (t - settle))
        for i in hip:
            d.ctrl[i] = base_hip + amp * s
        for i in knee:
            d.ctrl[i] = base_knee - amp * s
        d.ctrl[waist] = base_waist + waist_amp * s

    tr = run(model, data, int(duration / model.opt.timestep), target_fn, sample=0.05)
    good = tr["t"] > settle + 1.0 / freq
    grasp = (tr["left"][good] + tr["right"][good]) > 0
    grasp_frac = float(np.mean(grasp))
    com_swing = float(np.ptp(tr["com"][good, 0]))
    foot_swing = float(np.ptp(tr["foot"][good, 0]))
    lfrac = float(np.mean(tr["left"][good] > 0))
    rfrac = float(np.mean(tr["right"][good] > 0))
    fell = float(tr["pelvis"][-1, 2]) < z0 - 0.30
    verdict = (lfrac > 0.95) and (rfrac > 0.95) and (not fell) and (foot_swing > 0.10)
    return dict(
        name="M1.4 lower-body swing while hanging",
        passed=bool(verdict),
        amplitude_rad=amp,
        frequency_hz=freq,
        duration_s=duration,
        left_grasp_fraction=lfrac,
        right_grasp_fraction=rfrac,
        fell=fell,
        com_x_swing_peak_to_peak_m=com_swing,
        foot_x_swing_peak_to_peak_m=foot_swing,
        pelvis_z_end=float(tr["pelvis"][-1, 2]),
        torque_utilisation=torque_utilisation(model, data),
        trace={"t": tr["t"].tolist(), "com_x": tr["com"][:, 0].tolist(),
               "foot_x": tr["foot"][:, 0].tolist(), "pelvis_z": tr["pelvis"][:, 2].tolist()},
    )


CHECKS = {
    "M1.1": check_static_hang,
    "M1.2": check_single_hand,
    "M1.3": check_release_regrasp,
    "M1.4": check_leg_swing,
}


def grip_sweep(scales, **_) -> List[Dict]:
    """M1.2 verdict as a function of the finger torque scale."""
    out = []
    for scale in scales:
        params = bs.SceneParams(grip_torque_scale=scale)
        model, solved = bs.build_model(params)
        res = check_single_hand(model, mujoco.MjData(model))
        res["grip_torque_scale"] = scale
        tor = res["torque_utilisation"]
        print(f"[grip x{scale:>4}] {'PASS' if res['passed'] else 'FAIL'} "
              f"grasp={res['grasp_present_fraction']:.2f} fell={res['fell']} "
              f"worst={tor['worst_joint']}({tor['worst_util']:.2f})")
        out.append(res)
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def markdown(results: List[Dict], solved: Dict, params, sweep=None) -> str:
    lines = ["# M1 acceptance report (auto-generated)", "",
             f"Scene: Unitree G1 (43 actuated joints) + 5 capsule bars, "
             f"spacing {params.spacing:.2f} m, bar radius {params.bar_radius*1000:.0f} mm, "
             f"friction {params.bar_friction}.", "",
             f"- body mass: **{solved['info']['total_mass']:.2f} kg** "
             f"({solved['info']['total_mass']*9.81:.0f} N)",
             f"- bar position (x,z): ({solved['bar_xy'][0]:+.4f}, {solved['bar_xy'][1]:+.4f}) m",
             f"- CoM under bar: dx = {solved['dx']:+.5f} m",
             f"- grip torque scale: x{params.grip_torque_scale:g} "
             f"(Dex3 stock: 1.4/2.45 N*m -> {1.4*params.grip_torque_scale:.1f}/"
             f"{2.45*params.grip_torque_scale:.2f} N*m)",
             f"- wrist pitch/yaw locked by JOINT equality: {params.lock_wrists}", "",
             "| check | verdict | key numbers |", "|---|---|---|"]
    for r in results:
        if r["name"].startswith("M1.1"):
            num = (f"z drift {r['max_z_deviation_m']*1000:.1f} mm; "
                   f"grasp L/R {r['left_grasp_fraction']:.2f}/{r['right_grasp_fraction']:.2f}")
        elif r["name"].startswith("M1.2"):
            num = (f"grasp {r['grasp_present_fraction']:.2f}; fell={r['fell']}; "
                   f"worst {r['torque_utilisation']['worst_joint']} "
                   f"({r['torque_utilisation']['worst_util']:.2f} of limit)")
        elif r["name"].startswith("M1.3"):
            num = (f"released={r['released']}; regained={r['regained_contact']}; fell={r['fell']}")
        else:
            num = (f"leg swing {r['foot_x_swing_peak_to_peak_m']*1000:.0f} mm p-p; "
                   f"grasp L/R {r['left_grasp_fraction']:.2f}/{r['right_grasp_fraction']:.2f}; "
                   f"fell={r['fell']}")
        lines.append(f"| {r['name']} | **{'PASS' if r['passed'] else 'FAIL'}** | {num} |")
    if sweep:
        lines += ["", "## M1.2 vs finger torque scale", "",
                  "| grip x | grasp fraction | fell | worst joint (utilisation) |",
                  "|---|---|---|---|"]
        for s in sweep:
            tor = s["torque_utilisation"]
            lines.append(f"| x{s['grip_torque_scale']:g} | {s['grasp_present_fraction']:.2f} | "
                         f"{s['fell']} | {tor['worst_joint']} ({tor['worst_util']:.2f}) |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=None)
    ap.add_argument("--grip", default=bs.DEFAULT.grip, choices=list(bs.GRIP_PRESETS))
    ap.add_argument("--scene", default="xml", choices=["xml", "mjx", "build"],
                    help="xml = exported standalone MJCF, mjx = the MJX-compatible variant "
                         "(capsule proxies), build = rebuild with MjSpec")
    ap.add_argument("--grip-scale", type=float, default=bs.DEFAULT.grip_torque_scale)
    ap.add_argument("--grip-sweep", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "report.json"))
    ap.add_argument("--markdown", default=os.path.join(OUT_DIR, "report.md"))
    args = ap.parse_args()

    params = bs.SceneParams(grip=args.grip, grip_torque_scale=args.grip_scale)
    t0 = time.time()
    if args.scene in ("xml", "mjx"):
        path = bs.SCENE_XML if args.scene == "xml" else bs.SCENE_XML.replace(".xml", "_mjx.xml")
        model = bs.load_scene(path)
        meta = bs.load_meta(os.path.splitext(path)[0] + ".json")
        params = bs.SceneParams(**meta["params"])
        solved = {"info": {"total_mass": meta["total_mass"]},
                  "bar_xy": tuple(meta["bar_xy"]), "dx": meta["dx"],
                  "pose_kwargs": meta["pose_kwargs"]}
        print(f"[scene] loaded {os.path.basename(path)} (mujoco {mujoco.__version__})")
    else:
        model, solved = bs.build_model(params, verbose=True)
    data = mujoco.MjData(model)
    print(f"[scene] grip={params.grip} ({time.time()-t0:.1f}s)  mass={solved['info']['total_mass']:.2f} kg  "
          f"bar=({solved['bar_xy'][0]:+.3f},{solved['bar_xy'][1]:+.3f})  dx={solved['dx']:+.4f} m")
    rep_before = None
    if True:
        bs.reset_to(model, data)
        rep_before = bs.contact_report(model, data)
    print(f"[scene] keyframe contacts: L={rep_before['left']} R={rep_before['right']} "
          f"pen={rep_before['min_dist']*1000:+.2f} mm")

    names = [args.check] if args.check else list(CHECKS)
    results = []
    for name in names:
        res = CHECKS[name](model, data)
        results.append(res)
        extra = {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()
                 if k in ("max_z_deviation_m", "left_grasp_fraction", "right_grasp_fraction",
                          "grasp_present_fraction", "com_x_swing_peak_to_peak_m",
                          "foot_x_swing_peak_to_peak_m", "released", "regained_contact", "fell")}
        print(f"[{'PASS' if res['passed'] else 'FAIL'}] {res['name']}  " +
              "  ".join(f"{k}={v}" for k, v in extra.items()))
        if "torque_utilisation" in res:
            tu = res["torque_utilisation"]
            print(f"        torque utilisation: arm={tu['arm']:.2f} hand={tu['hand']:.2f} "
                  f"leg={tu['leg']:.2f} waist={tu['waist']:.2f} worst={tu['worst_joint']}")

    sweep = grip_sweep((1.0, 5.0, 10.0, 20.0)) if args.grip_sweep else None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(params=dataclasses.asdict(params), solved_bar=list(solved["bar_xy"]),
                       total_mass=solved["info"]["total_mass"],
                       results=results, grip_sweep=sweep), fh, indent=2, default=float)
    with open(args.markdown, "w") as fh:
        fh.write(markdown(results, solved, params, sweep))
    print(f"[report] {args.out}\n[report] {args.markdown}")
    ok = all(r["passed"] for r in results)
    print(f"[summary] {'ALL PASS' if ok else 'SOME FAILED: ' + ', '.join(r['name'] for r in results if not r['passed'])}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
