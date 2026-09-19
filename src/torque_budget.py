"""Static torque budget of the hanging G1 (M0 deliverable).

Two views of the same question -- *which joints are the binding constraint while
the robot hangs from the bar?*

1. **Simulated**: after settling at the ``hang`` keyframe, read
   ``data.qfrc_actuator`` (the force each position servo actually applies) and
   compare it with ``jnt_actfrcrange`` (the Unitree joint torque limit).
2. **Analytic**: for every arm/waist joint, the static moment arm is the
   horizontal distance from the joint centre of rotation to the vertical line
   through the bar, and the load is the weight carried by that chain:

       tau = F * |x_joint - x_bar|

   In a two-hand hang each arm carries 1/2 of the body weight, in a single-hand
   hang the whole weight, and the waist carries the lower body (pelvis + legs).

Run::

    .venv/bin/python src/torque_budget.py
"""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "assets", "g1_brachiation"))

import build_scene as bs  # noqa: E402

ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]
WAIST_JOINTS = ["waist_yaw", "waist_roll", "waist_pitch"]
HAND_FRACTIONS = {"thumb_0": 2.45, "thumb_1": 1.4, "thumb_2": 1.4,
                  "middle_0": 1.4, "middle_1": 1.4, "index_0": 1.4, "index_1": 1.4}


def settled_state(params: bs.SceneParams):
    model, solved = bs.build_model(params)
    data = mujoco.MjData(model)
    bs.reset_to(model, data)
    data.ctrl[:] = bs.DEFAULT and model.key_ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hang")].copy()
    for _ in range(200):
        mujoco.mj_step(model, data)
    return model, solved, data


def simulated_table(model: mujoco.MjModel, data: mujoco.MjData):
    rows = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name or not model.jnt_actfrclimited[jid]:
            continue
        limit = max(abs(model.jnt_actfrcrange[jid]))
        if limit <= 0:
            continue
        tau = float(data.qfrc_actuator[model.jnt_dofadr[jid]])
        rows.append((abs(tau) / limit, name, tau, limit))
    rows.sort(reverse=True)
    return rows


def analytic_table(model: mujoco.MjModel, solved: dict, total_mass: float):
    """Static moment-arm analysis for the two-hand and single-hand hang."""
    W = total_mass * 9.81
    bar_x = solved["bar_xy"][0]
    lower_mass = _subtree_mass(model, "left_hip_pitch_link") + _subtree_mass(model, "right_hip_pitch_link") + _subtree_mass(model, "pelvis") * 0 + _pelvis_mass(model)
    rows = []
    for side in ("left", "right"):
        for joint in ARM_JOINTS:
            body = _body_for_joint(model, f"{side}_{joint}_joint")
            if body is None:
                continue
            x = float(model.body_pos[body][0]) if False else None
            rows.append((side, joint, body))
    return W, lower_mass, rows


def _pelvis_mass(model):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return float(model.body_mass[bid])


def _subtree_mass(model, body_name: str) -> float:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    total = 0.0
    root = int(model.body_rootid[bid])
    for b in range(model.nbody):
        if int(model.body_rootid[b]) == root and b >= bid:
            # approximate: all bodies whose ancestor chain contains bid
            p = b
            while p != 0:
                if p == bid:
                    total += float(model.body_mass[b])
                    break
                p = int(model.body_parentid[p])
    return total


def _body_for_joint(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    return int(model.jnt_bodyid[jid])


def main() -> int:
    params = bs.SceneParams()
    model, solved, data = settled_state(params)
    total = solved["info"]["total_mass"]
    W = total * 9.81
    bar_x, bar_z = solved["bar_xy"]
    print(f"body mass {total:.2f} kg   weight {W:.1f} N   bar x={bar_x:+.4f} z={bar_z:+.4f}")
    print(f"CoM     {np.round(data.subtree_com[0], 4)}   dx(CoM-bar)="
          f"{data.subtree_com[0][0] - bar_x:+.4f} m\n")

    print("=== simulated actuator utilisation at the settled two-hand hang ===")
    print(f"{'#':>3} {'joint':<28}{'tau (N*m)':>11}{'limit':>9}{'util':>7}")
    for i, (util, name, tau, limit) in enumerate(simulated_table(model, data)[:12]):
        print(f"{i:>3} {name:<28}{tau:>11.2f}{limit:>9.2f}{util:>7.2f}")

    print("\n=== analytic static moment arm (load line through the bar) ===")
    print(f"{'chain':<34}{'x_joint':>9}{'|dx|':>8}{'load N':>9}{'tau':>9}{'limit':>8}{'util':>7}")
    rows = []
    for side in ("left", "right"):
        for joint in ARM_JOINTS:
            body = _body_for_joint(model, f"{side}_{joint}_joint")
            if body is None:
                continue
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")
            limit = max(abs(model.jnt_actfrcrange[jid])) if model.jnt_actfrclimited[jid] else float("inf")
            x = float(data.xpos[body][0])
            dx = abs(x - bar_x)
            tau_two = 0.5 * W * dx
            rows.append((tau_two / limit if limit else 0, f"{side}.{joint}", x, dx, 0.5 * W, tau_two, limit))
    for util, name, x, dx, load, tau, limit in sorted(rows, reverse=True)[:12]:
        print(f"{name:<34}{x:>9.4f}{dx:>8.4f}{load:>9.1f}{tau:>9.2f}{limit:>8.1f}{util:>7.2f}")

    print("\n=== single-hand hang (one chain carries the whole weight) ===")
    worst = []
    for side in ("left",):
        for joint in ARM_JOINTS:
            body = _body_for_joint(model, f"{side}_{joint}_joint")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")
            limit = max(abs(model.jnt_actfrcrange[jid])) if model.jnt_actfrclimited[jid] else float("inf")
            x = float(data.xpos[body][0])
            dx = abs(x - bar_x)
            tau = W * dx
            worst.append((tau / limit if limit else 0, f"{side}.{joint}", dx, tau, limit))
    print(f"{'chain':<34}{'|dx|':>8}{'tau':>9}{'limit':>8}{'util':>7}  max |dx| allowed at limit")
    for util, name, dx, tau, limit in sorted(worst, reverse=True):
        allowed = limit / W * 1000 if W else 0
        print(f"{name:<34}{dx:>8.4f}{tau:>9.2f}{limit:>8.1f}{util:>7.2f}   {allowed:6.1f} mm")

    lower = _subtree_mass(model, "left_hip_pitch_link") + _subtree_mass(model, "right_hip_pitch_link") + _pelvis_mass(model)
    d = abs(float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link")][0] - bar_x))
    print(f"\nlower body (pelvis + legs) mass {lower:.2f} kg -> waist load {lower*9.81:.1f} N; "
          f"waist_pitch limit 50 N*m allows a moment arm of {50/(lower*9.81)*1000:.0f} mm "
          f"(hip->knee x-offset now {d*1000:.0f} mm)")

    print("\n=== Dex3 finger budget (why the grip scale matters) ===")
    print(f"{'finger joint':<16}{'stock limit':>12}{'x{scale}'.format(scale=params.grip_torque_scale):>10}"
          f"{'@53 N/finger':>14}{'@105 N/finger':>15}")
    for joint, limit in HAND_FRACTIONS.items():
        lever = 0.04  # m, knuckle-to-bar moment arm (order of magnitude)
        print(f"{joint:<16}{limit:>12.2f}{limit*params.grip_torque_scale:>10.2f}"
              f"{53*lever:>14.2f}{105*lever:>15.2f}")
    print("(moment arm 40 mm; 53 N = 1/3 of half the body weight, 105 N = 1/3 of full weight)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
