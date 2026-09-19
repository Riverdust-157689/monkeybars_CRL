"""M0: build the Unitree G1 monkey-bar (brachiation) MuJoCo scene.

The scene is generated programmatically from the upstream Menagerie
``g1_with_hands.xml`` (43 actuated joints = 12 legs + 3 waist + 14 arms + 14
Dex3-1 hand joints) so that every deviation from the stock robot is explicit and
reproducible:

* ``grip_torque_scale`` -- multiplies ``actfrcrange`` of the 14 Dex3 joints.
  The stock finger joints are limited to 1.4 / 2.45 N*m, which cannot carry the
  34.4 kg body (see ``docs/M0_M1_报告.md``).  Raising this is a *hardware
  layer* deviation and is recorded as such.
* ``lock_wrists``       -- JOINT equalities holding wrist pitch/yaw at 0 (they
  are 5 N*m actuators sitting on the suspension load path).
* bars                 -- capsule geoms welded to the world, condim=4 with high
  friction, placed at the *centre of the finger cage* of the closed hand.
* keyframe "hang"      -- a **settled** state: the hands start open around the
  bar and are closed over 0.4 s, then the state is snapshotted.  Posing the
  closed hand kinematically would bury the bar 20-40 mm inside the fingers and
  the resulting contact impulse launches the robot.

Usage::

    import build_scene as bs
    model, info = bs.build_model(bs.SceneParams())
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ROBOT_XML = os.path.join(HERE, "menagerie", "unitree_g1", "g1_with_hands.xml")

#: Exported, standalone scene (what the RL runtime loads).  Building the scene
#: needs a recent MuJoCo (MjSpec); running it does not, so the training stack can
#: be pinned to JaxGCRL's own MuJoCo/MJX version without touching the builder.
SCENE_XML = os.path.join(HERE, "scene_bars.xml")
SCENE_META = os.path.join(HERE, "scene_bars.json")


# --------------------------------------------------------------------------- #
# Pose definition
# --------------------------------------------------------------------------- #

#: Fully-closed target of the left hand (the right hand mirrors the sign, since
#: its joint ranges are mirrored).  Values are the range extreme in the curling
#: direction, taken from the Menagerie model.
FINGER_CLOSED_LEFT = {
    "thumb_0": 1.0472,
    "thumb_1": 1.0472,
    "thumb_2": 1.74533,
    "middle_0": -1.5708,
    "middle_1": -1.74533,
    "index_0": -1.5708,
    "index_1": -1.74533,
}

#: Fraction of the full curl commanded when the hand is "closed".
FINGER_CLOSURE = {
    "thumb_0": 0.85,
    "thumb_1": 0.95,
    "thumb_2": 0.95,
    "middle_0": 0.85,
    "middle_1": 0.85,
    "index_0": 0.85,
    "index_1": 0.85,
}

#: Grip presets.  ``forward`` = palm faces the same way as the face (+x, the
#: overhand / "正手" grip); ``backhand`` = palm faces away from the face (the
#: earlier M1 configuration, kept for ablation and reproducibility).
#: ``cage_offset`` is the bar centre relative to the mid-point of the
#: index/middle knuckles; each preset was calibrated by a settle-and-hold scan.
GRIP_PRESETS = {
    "forward": {          # 正手: palm faces +x, same as the face; passes M1.1-M1.4 at grip x5
        "shoulder_pitch": -2.75,
        "shoulder_roll": 1.50,
        "shoulder_yaw": -1.20,
        "elbow": 0.50,
        "wrist_roll": 0.10,
        "wrist_pitch": 0.0,
        "wrist_yaw": 0.0,
        "cage_offset": (0.015, -0.036),
    },
    "backhand": {
        "shoulder_pitch": -2.60,
        "shoulder_roll": 0.20,
        "shoulder_yaw": 0.00,
        "elbow": 0.15,
        "wrist_roll": -1.40,
        "wrist_pitch": 0.0,
        "wrist_yaw": 0.0,
        "cage_offset": (0.000, 0.000),
    },
}

#: Default arm preset used by :func:`pose_from_params`.
HANG_ARM = {k: v for k, v in GRIP_PRESETS["forward"].items() if k != "cage_offset"}

#: Legs held in a tuck (v1 "B - fixed servo" allocation from the plan).
HANG_LEGS = {
    "hip_pitch": -0.60,
    "hip_roll": 0.05,
    "hip_yaw": 0.0,
    "knee": 1.20,
    "ankle_pitch": -0.60,
    "ankle_roll": 0.0,
}

WAIST = {"waist_yaw": 0.0, "waist_roll": 0.0, "waist_pitch": 0.0}

#: Bar centre relative to the mid-point of the index/middle knuckles, i.e. the
#: centre of the finger cage (calibrated by a settle-and-hold scan, see M0/M1 report).
CAGE_OFFSET = GRIP_PRESETS["forward"]["cage_offset"]

#: The MJX proxy model (capsule body colliders) has slightly different contact
#: geometry around the wrist, so its bar seat is calibrated separately.
CAGE_OFFSET_MJX = (0.030, -0.036)

#: Joints whose left/right values are mirrored in sign.
MIRROR = ("roll", "yaw")


def _mirror_value(side: str, joint: str, value: float) -> float:
    if side == "right" and any(joint.endswith(m) or f"_{m}_" in joint for m in MIRROR):
        return -value
    return value


def pose_from_params(
    *,
    grip: str = "forward",
    shoulder_pitch: Optional[float] = None,
    shoulder_roll: Optional[float] = None,
    shoulder_yaw: Optional[float] = None,
    elbow: Optional[float] = None,
    wrist_roll: Optional[float] = None,
    hip_pitch: Optional[float] = None,
    knee: Optional[float] = None,
    ankle_pitch: Optional[float] = None,
    waist_pitch: float = 0.0,
) -> Dict[str, float]:
    """Return ``{joint_name: value}`` for the symmetric hang pose.

    Unset arguments fall back to ``GRIP_PRESETS[grip]`` (arms) and ``HANG_LEGS``.
    """
    arm = {k: v for k, v in GRIP_PRESETS[grip].items() if k != "cage_offset"}
    for key, value in (("shoulder_pitch", shoulder_pitch), ("shoulder_roll", shoulder_roll),
                       ("shoulder_yaw", shoulder_yaw), ("elbow", elbow),
                       ("wrist_roll", wrist_roll)):
        if value is not None:
            arm[key] = value
    legs = dict(HANG_LEGS)
    if hip_pitch is not None:
        legs["hip_pitch"] = hip_pitch
    if knee is not None:
        legs["knee"] = knee
    if ankle_pitch is not None:
        legs["ankle_pitch"] = ankle_pitch

    pose: Dict[str, float] = {}
    for side in ("left", "right"):
        for joint, value in arm.items():
            pose[f"{side}_{joint}_joint"] = _mirror_value(side, joint, value)
        for joint, value in legs.items():
            pose[f"{side}_{joint}_joint"] = _mirror_value(side, joint, value)
    for joint, value in WAIST.items():
        pose[f"{joint}_joint"] = waist_pitch if joint == "waist_pitch" else value
    return pose


def finger_pose(closure: float = 1.0) -> Dict[str, float]:
    """Finger targets: ``closure`` scales the curl from open (0) to closed (1)."""
    pose = {}
    for side in ("left", "right"):
        sign = 1.0 if side == "left" else -1.0
        for joint, limit in FINGER_CLOSED_LEFT.items():
            pose[f"{side}_hand_{joint}_joint"] = sign * FINGER_CLOSURE[joint] * closure * limit
    return pose


# --------------------------------------------------------------------------- #
# Scene parameters
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class SceneParams:
    grip: str = "forward"          # "forward" (正手) or "backhand" (反手)
    grip_torque_scale: float = 10.0
    bar_radius: float = 0.025
    cage_offset: Optional[Tuple[float, float]] = None   # None -> preset value
    bar_half_length: float = 0.45
    bar_friction: float = 1.4
    bar_condim: int = 4
    n_bars: int = 5
    spacing: float = 0.40
    lock_wrists: bool = True
    sparse_collision: bool = False
    hand_kp: float = 0.0   # 0 = keep the stock kp=500 / dampratio=1 servo
    hand_kv: float = 1.5
    timestep: float = 0.002
    close_frac: float = 0.95      # finger curl commanded at the grasp keyframe
    close_time: float = 0.5       # duration of the closing ramp (zero-g phase)
    gravity_free_time: float = 0.7
    gravity_ramp_time: float = 1.5
    settle_time: float = 0.6      # tail after gravity is fully on
    keyframe: bool = True
    mjx_compat: bool = True       # replace features MJX-JAX cannot compile
    mjx_keep_mesh: Tuple[str, ...] = ("_hand_",)   # bodies that keep mesh colliders
    mjx_strip_visual: bool = True
    mjx_maxhullvert: int = 0      # >0 caps kept mesh colliders' convex hull size

    def __post_init__(self) -> None:
        if self.grip not in GRIP_PRESETS:
            raise ValueError(f"unknown grip {self.grip!r}; use one of {list(GRIP_PRESETS)}")
        if self.cage_offset is None:
            self.cage_offset = CAGE_OFFSET_MJX if self.mjx_compat else GRIP_PRESETS[self.grip]["cage_offset"]


DEFAULT = SceneParams()
_MODEL_CACHE: Dict[tuple, mujoco.MjModel] = {}


# --------------------------------------------------------------------------- #
# Spec construction
# --------------------------------------------------------------------------- #


def _load_spec(params: SceneParams) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(ROBOT_XML)
    option = spec.option
    option.timestep = params.timestep
    option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    option.iterations = 50
    option.ls_iterations = 20
    option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    option.gravity = [0.0, 0.0, -9.81]
    return spec


def _apply_grip_torque(spec: mujoco.MjSpec, scale: float) -> None:
    if scale == 1.0:
        return
    for joint in spec.joints:
        if "_hand_" in joint.name:
            joint.actfrcrange = np.array(joint.actfrcrange, dtype=float) * scale
            joint.actfrclimited = True


def _lock_wrists(spec: mujoco.MjSpec) -> None:
    for side in ("left", "right"):
        for joint in ("wrist_pitch", "wrist_yaw"):
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.name = f"{side}_{joint}_lock"
            eq.name1 = f"{side}_{joint}_joint"
            eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
            eq.solref = [0.002, 1.0]
            eq.solimp = [0.99, 0.999, 0.0001, 0.5, 2.0]
            # polycoef defaults to "0 0 0 0 0" -> constrains q == 0


def _add_bars(spec: mujoco.MjSpec, params: SceneParams, bar_x: float, bar_z: float) -> None:
    quat_y = [float(np.sqrt(0.5)), float(np.sqrt(0.5)), 0.0, 0.0]  # capsule +Z -> world Y
    for i in range(params.n_bars):
        body = spec.worldbody.add_body()
        body.name = f"bar{i}"
        body.pos = [bar_x + i * params.spacing, 0.0, bar_z]
        geom = body.add_geom()
        geom.name = f"bar{i}_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        geom.size = [params.bar_radius, params.bar_half_length, 0.0]
        geom.quat = quat_y
        geom.condim = params.bar_condim
        geom.friction = [params.bar_friction, 0.02, 0.0001]
        geom.rgba = [0.35, 0.35, 0.38, 1.0]
        geom.density = 0.0  # static geometry
        geom.contype = 0b001
        geom.conaffinity = 0b110


def _set_hand_gains(spec: mujoco.MjSpec, kp: float, kv: float) -> None:
    """Soft impedance for the Dex3 fingers (the stock kp=500 is very stiff)."""
    if kp <= 0:
        return
    for act in spec.actuators:
        if "_hand_" in act.name:
            gain = list(act.gainprm)
            bias = list(act.biasprm)
            gain[0] = kp
            bias[0] = 0.0
            bias[1] = -kp
            bias[2] = -kv
            act.gainprm = gain
            act.biasprm = bias


def _set_collision_bits(spec: mujoco.MjSpec) -> None:
    """Sparse contact graph: only hand<->bar, body<->bar, hand<->body.

    The stock model has automatic full self-collision.  With a closed Dex3 hand
    that makes the finger links collide with each other and with the palm, which
    jams the curl (the proximal joints get pushed back to their open limit) and
    makes grasp formation a coin flip.  Bit masks (MuJoCo rule:
    ``(a.contype & b.conaffinity) | (b.contype & a.conaffinity)``):

      bars      contype=0b001 conaffinity=0b110
      hands     contype=0b010 conaffinity=0b101
      the rest  contype=0b100 conaffinity=0b011
    """
    for body in spec.bodies:
        is_hand = "_hand_" in body.name
        for geom in body.geoms:
            if geom.contype == 0 and geom.conaffinity == 0:
                continue  # visual-only geom
            if is_hand:
                geom.contype, geom.conaffinity = 0b010, 0b101
            else:
                geom.contype, geom.conaffinity = 0b100, 0b011


def _robot_only(params: SceneParams) -> mujoco.MjModel:
    key = ("robot", params.grip_torque_scale, params.lock_wrists, params.timestep,
           params.sparse_collision, params.hand_kp, params.hand_kv)
    if key not in _MODEL_CACHE:
        spec = _load_spec(params)
        _apply_grip_torque(spec, params.grip_torque_scale)
        _set_hand_gains(spec, params.hand_kp, params.hand_kv)
        if params.sparse_collision:
            _set_collision_bits(spec)
        if params.lock_wrists:
            _lock_wrists(spec)
        _MODEL_CACHE[key] = spec.compile()
    return _MODEL_CACHE[key]


def _apply_pose(model: mujoco.MjModel, data: mujoco.MjData, values: Dict[str, float],
                base: Optional[np.ndarray] = None) -> None:
    data.qpos[:] = model.qpos0 if base is None else base
    for name, value in values.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(name)
        lo, hi = model.jnt_range[jid]
        data.qpos[model.jnt_qposadr[jid]] = float(np.clip(value, lo, hi))
    mujoco.mj_forward(model, data)


def _ctrl_from_qpos(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=float)
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        ctrl[i] = qpos[model.jnt_qposadr[jid]]
    return ctrl


def probe_grip(params: SceneParams, pose: Dict[str, float], closure: float = 1.0):
    """Forward kinematics of the hanging pose (fingers at ``closure``)."""
    model = _robot_only(params)
    data = mujoco.MjData(model)
    _apply_pose(model, data, {**pose, **finger_pose(closure)})

    def bid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    out = {}
    for side in ("left", "right"):
        tips = [data.xpos[bid(f"{side}_hand_{b}_link")] for b in ("thumb_2", "middle_1", "index_1")]
        knuckle = 0.5 * (data.xpos[bid(f"{side}_hand_index_0_link")]
                         + data.xpos[bid(f"{side}_hand_middle_0_link")])
        out[side] = dict(
            tip=np.mean(tips, axis=0),
            knuckle=knuckle,
            wrist=data.xpos[bid(f"{side}_wrist_roll_link")],
            shoulder=data.xpos[bid(f"{side}_shoulder_pitch_link")],
            elbow=data.xpos[bid(f"{side}_elbow_link")],
        )
    out["com"] = data.subtree_com[0].copy()
    out["total_mass"] = float(model.body_mass.sum())
    return out


def grip_point(params: SceneParams, info) -> np.ndarray:
    knuckle = 0.5 * (info["left"]["knuckle"] + info["right"]["knuckle"])
    return knuckle + np.array([params.cage_offset[0], 0.0, params.cage_offset[1]])


def _bar_xz(params: SceneParams, pose_kwargs) -> Tuple[float, float, float]:
    """Return ``(bar_x, bar_z, dx_balance)`` for the given pose parameters."""
    pose = pose_from_params(**pose_kwargs)
    info = probe_grip(params, pose, closure=1.0)
    grip = grip_point(params, info)
    return float(grip[0]), float(grip[2]), float(grip[0] - info["com"][0])


def _solve_hip_pitch(params: SceneParams, pose_kwargs, lo=-1.30, hi=1.30, iters=40) -> float:
    """Bisection on ``hip_pitch`` so that the CoM sits under the bar."""

    def dx(hp):
        kw = dict(pose_kwargs)
        kw["hip_pitch"] = hp
        return _bar_xz(params, kw)[2]

    f_lo, f_hi = dx(lo), dx(hi)
    if f_lo * f_hi > 0:
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = dx(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def solve_hang(params: SceneParams = DEFAULT, **pose_kwargs):
    """Solve the hang pose: legs tucked, CoM under the bar, hands on the bar."""
    pose_kwargs.setdefault("grip", params.grip)
    if pose_kwargs.get("hip_pitch") is None:
        pose_kwargs["hip_pitch"] = _solve_hip_pitch(params, pose_kwargs)
    pose = pose_from_params(**pose_kwargs)
    info = probe_grip(params, pose, closure=1.0)
    grip = grip_point(params, info)
    return dict(
        pose=pose,
        pose_kwargs=pose_kwargs,
        fingers=finger_pose(1.0),
        info=info,
        grip=grip,
        bar_xy=(float(grip[0]), float(grip[2])),
        dx=float(grip[0] - info["com"][0]),
    )


def _settle(model: mujoco.MjModel, params: SceneParams, pose: Dict[str, float]):
    """Close the hands around the bar and let the state settle; return qpos."""
    data = mujoco.MjData(model)
    _apply_pose(model, data, {**pose, **finger_pose(0.0)})

    target = np.zeros(model.nu, dtype=float)
    for name, value in {**pose, **finger_pose(params.close_frac)}.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = int(np.flatnonzero(model.actuator_trnid[:, 0] == jid)[0])
        lo, hi = model.jnt_range[jid]
        target[aid] = float(np.clip(value, lo, hi))

    open_ctrl = target.copy()
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if "_hand_" in nm:
            open_ctrl[i] = 0.0

    dt = model.opt.timestep
    gravity = model.opt.gravity.copy()
    # Phase 1: close the fingers with gravity switched off, so the robot is
    # supported by its own posture instead of free-falling while the hand is
    # still open (that made grasp formation a coin flip).
    model.opt.gravity[:] = 0.0
    n_close = max(1, int(params.close_time / dt))
    for step in range(max(1, int(params.gravity_free_time / dt))):
        frac = min(1.0, (step + 1) / n_close)
        data.ctrl[:] = open_ctrl + frac * (target - open_ctrl)
        mujoco.mj_step(model, data)
    # Phase 2: ramp gravity on, transferring the load to the grasp.
    n_ramp = max(1, int(params.gravity_ramp_time / dt))
    for step in range(n_ramp):
        data.ctrl[:] = target
        model.opt.gravity[:] = gravity * min(1.0, (step + 1) / n_ramp)
        mujoco.mj_step(model, data)
    # Phase 3: settle with full gravity.
    for _ in range(max(1, int(params.settle_time / dt))):
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
    model.opt.gravity[:] = gravity
    return data.qpos.copy(), data


def build_spec(params: SceneParams = DEFAULT, verbose: bool = False, **pose_kwargs) -> mujoco.MjSpec:
    solved = solve_hang(params, **pose_kwargs)
    pose = solved["pose"]
    bar_x, bar_z = solved["bar_xy"]

    spec = _load_spec(params)
    _apply_grip_torque(spec, params.grip_torque_scale)
    _set_hand_gains(spec, params.hand_kp, params.hand_kv)
    if params.sparse_collision:
        _set_collision_bits(spec)
    if params.lock_wrists:
        _lock_wrists(spec)
    if params.mjx_compat:
        _make_mjx_compatible(spec)
        if params.mjx_strip_visual:
            print(f"[mjx] stripped {_strip_visual(spec)} visual geoms")
        print(f"[mjx] fitted {_fit_primitive_proxies(spec, params.mjx_keep_mesh)} capsule proxies")
        if params.mjx_maxhullvert:
            for mesh in spec.meshes:
                mesh.maxhullvert = params.mjx_maxhullvert
            print(f"[mjx] capped maxhullvert at {params.mjx_maxhullvert}")
    _add_bars(spec, params, bar_x, bar_z)

    key = spec.add_key()
    key.name = "hang"
    base = spec.compile()
    key.qpos = np.array(base.qpos0, dtype=float)
    key.ctrl = np.zeros(base.nu)

    if params.keyframe:
        qpos, data = _settle(base, params, pose)
        # Relaxed hang: command the *settled* posture everywhere, so the servos
        # do not fight the closed kinematic loop (both hands on one rigid bar),
        # and only the fingers keep a squeeze beyond the contact point so that
        # the grasp carries a real force.
        ctrl = _ctrl_from_qpos(base, qpos)
        for name, value in {**pose, **finger_pose(params.close_frac)}.items():
            if "_hand_" not in name:
                continue
            jid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = int(np.flatnonzero(base.actuator_trnid[:, 0] == jid)[0])
            lo, hi = base.jnt_range[jid]
            ctrl[aid] = float(np.clip(value, lo, hi))
        key.qpos = qpos
        key.ctrl = ctrl
        if verbose:
            com = data.subtree_com[0]
            print(f"[build] settled |qvel|={np.linalg.norm(data.qvel):.4f} "
                  f"pelvis_z={qpos[2]:.4f} com_x={com[0]:+.4f} bar_x={bar_x:+.4f}")
            pretty = {k: (round(v, 4) if isinstance(v, (int, float)) else v)
                      for k, v in solved["pose_kwargs"].items()}
            print(f"[build] pose_kwargs={pretty}")
    return spec


def build_model(params: SceneParams = DEFAULT, verbose: bool = False, **pose_kwargs):
    """Return ``(model, solved)`` with the solved bar placement, pose and keyframe."""
    solved = solve_hang(params, **pose_kwargs)
    spec = build_spec(params, verbose=verbose, **solved["pose_kwargs"])
    return spec.compile(), solved


# --------------------------------------------------------------------------- #
# Runtime helpers
# --------------------------------------------------------------------------- #


def reset_to(model: mujoco.MjModel, data: mujoco.MjData, bar_index: int = 0,
             dxy=(0.0, 0.0), key: str = "hang") -> None:
    """Reset to a keyframe and (optionally) shift onto another bar."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
    if kid < 0:
        raise KeyError(key)
    mujoco.mj_resetDataKeyframe(model, data, kid)
    if bar_index != 0 or dxy != (0.0, 0.0):
        data.qpos[0] = model.key_qpos[kid][0] + bar_index * bar_spacing(model) + dxy[0]
        data.qpos[1] = model.key_qpos[kid][1] + dxy[1]
    mujoco.mj_forward(model, data)


def bar_spacing(model: mujoco.MjModel) -> float:
    b0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bar0")
    b1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bar1")
    return float(model.body_pos[b1][0] - model.body_pos[b0][0])


def bar_geom_ids(model: mujoco.MjModel):
    ids = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith("bar") and name.endswith("_geom"):
            ids.append(i)
    return ids


def hand_geom_ids(model: mujoco.MjModel, side: str = "left"):
    ids = []
    for i in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or ""
        if f"{side}_hand" in body:
            ids.append(i)
    return ids


def contact_report(model: mujoco.MjModel, data: mujoco.MjData):
    """Per-hand contact summary against the bars."""
    bars = set(bar_geom_ids(model))
    out = {"n_bar_contacts": 0, "min_dist": 0.0, "max_force": 0.0,
           "left": 0, "right": 0, "bars_touched": set()}
    f = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if not pair & bars:
            continue
        other = int(c.geom2) if int(c.geom1) in bars else int(c.geom1)
        bar = int(c.geom1) if int(c.geom1) in bars else int(c.geom2)
        out["n_bar_contacts"] += 1
        out["min_dist"] = min(out["min_dist"], float(c.dist))
        out["bars_touched"].add(bar)
        mujoco.mj_contactForce(model, data, i, f)
        out["max_force"] = max(out["max_force"], float(np.linalg.norm(f[:3])))
        for side in ("left", "right"):
            if other in hand_geom_ids(model, side):
                out[side] += 1
    return out


# --------------------------------------------------------------------------- #
# Export / load (decouples the builder from the RL runtime MuJoCo version)
# --------------------------------------------------------------------------- #


_AXIS_QUAT = {
    2: [1.0, 0.0, 0.0, 0.0],            # capsule +Z stays +Z
    0: [0.70710678, 0.0, 0.70710678, 0.0],   # +Z -> +X
    1: [0.70710678, -0.70710678, 0.0, 0.0],  # +Z -> +Y
}


def _quat_mul(a, b):
    aw, ax, ay, az = np.asarray(a, dtype=float)
    bw, bx, by, bz = np.asarray(b, dtype=float)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_rotate(q, v):
    q = np.asarray(q, dtype=float)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, dtype=float)


def _fit_primitive_proxies(spec: mujoco.MjSpec, keep_substrings=("_hand_",)) -> int:
    """Replace body mesh collision geoms with capsule proxies fitted to their AABB.

    MJX-JAX compiles every collision mesh into SAT/BVH structures; the G1 with
    its Dex3 hands has 86 collision meshes and exceeded available RAM before it
    could even build the model.  Capsule proxies are MJX-native and cheap.
    Bodies whose name contains one of ``keep_substrings`` keep their meshes (the
    grasp contact geometry lives in the fingers).

    The AABB comes from the compiled model (``geom_aabb``, geom-local frame), and
    compiled geoms are mapped back to spec geoms by body name + order, which
    MuJoCo preserves.
    """
    model = spec.compile()
    body_geom_count = {}
    compiled_by_body = {}
    for g in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
        compiled_by_body.setdefault(bname, []).append(g)
    n = 0
    for body in spec.bodies:
        gids = compiled_by_body.get(body.name)
        if gids is None:
            continue
        spec_geoms = list(body.geoms)
        if len(spec_geoms) != len(gids):
            raise RuntimeError(f"geom mapping mismatch on body {body.name}: "
                               f"{len(spec_geoms)} spec vs {len(gids)} compiled")
        if any(k in body.name for k in keep_substrings):
            continue
        for geom, gid in zip(spec_geoms, gids):
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_contype[gid] == 0:
                continue
            lo, hi = model.geom_aabb[gid][:3], model.geom_aabb[gid][3:]
            center, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            axis = int(np.argmax(half))
            radius = float(np.min(np.delete(half, axis)))
            quat = np.asarray(model.geom_quat[gid], dtype=float)
            offset = _quat_rotate(quat, center)
            new_quat = _quat_mul(quat, np.asarray(_AXIS_QUAT[axis], dtype=float))
            geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            geom.meshname = ""
            geom.pos = [float(model.geom_pos[gid][0] + offset[0]),
                        float(model.geom_pos[gid][1] + offset[1]),
                        float(model.geom_pos[gid][2] + offset[2])]
            geom.quat = [float(v) for v in new_quat]
            geom.size = [max(radius, 5e-3), max(float(half[axis]) - radius, 5e-3), 0.0]
            n += 1
    return n


def _strip_visual(spec: mujoco.MjSpec) -> int:
    """Remove visual-only geoms (MJX does not render; they cost memory)."""
    doomed = [geom for body in spec.bodies for geom in body.geoms
              if geom.contype == 0 and geom.conaffinity == 0]
    for geom in doomed:
        spec.delete(geom)
    return len(doomed)


def _make_mjx_compatible(spec: mujoco.MjSpec) -> None:
    """Replace features MJX-JAX cannot compile.

    MJX-JAX refuses to load a model containing cylinder<->mesh collision pairs
    ("collisions between (SPHERE, BOX, MESH, HFIELD) and CYLINDER" are
    unimplemented).  The G1 model uses 4 cylinder colliders for the shoulder
    joint housings; turning them into capsules (same radius / half length) is
    geometrically almost identical and MJX-safe.
    """
    n = 0
    for body in spec.bodies:
        for geom in body.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                radius, half = float(geom.size[0]), float(geom.size[1])
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                geom.size = [radius, half, 0.0]
                n += 1
    if n:
        print(f"[mjx] converted {n} cylinder colliders to capsules")


def _downgrade_xml(xml: str) -> str:
    """Drop elements/attributes that newer MuJoCo emits but older versions reject.

    We build with a recent MuJoCo (MjSpec iteration API) and train with JaxGCRL's
    pinned MuJoCo 3.2.7, so the exported scene must be loadable by both.  The
    only offender so far is the auto-generated head light (its ``type``
    attribute is newer); lights are purely visual and MuJoCo re-adds a default
    one when none is present.
    """
    import re

    xml = re.sub(r"\s*<light\b[^>]*/>", "", xml)
    xml = re.sub(r"\s*<light\b[^>]*>.*?</light>", "", xml, flags=re.S)
    return xml


def export_scene(params: SceneParams = DEFAULT, path: str = SCENE_XML,
                 meta_path: str = None, verbose: bool = True, **pose_kwargs):
    """Compile the scene and write a standalone MJCF + metadata sidecar."""
    import json

    if meta_path is None:
        meta_path = os.path.splitext(path)[0] + ".json"
    solved = solve_hang(params, **pose_kwargs)
    spec = build_spec(params, verbose=verbose, **solved["pose_kwargs"])
    # to_xml needs an absolute meshdir; rewrite it to a path relative to this
    # file so the exported scene stays portable.
    abs_meshdir = os.path.join(HERE, "menagerie", "unitree_g1", "assets")
    spec.compiler.meshdir = abs_meshdir
    xml = spec.to_xml()
    xml = xml.replace(abs_meshdir + "/", "menagerie/unitree_g1/assets/")
    xml = xml.replace(abs_meshdir, "menagerie/unitree_g1/assets")
    xml = _downgrade_xml(xml)
    with open(path, "w") as fh:
        fh.write(xml)
    meta = dict(
        grip=params.grip,
        bar_xy=[float(solved["bar_xy"][0]), float(solved["bar_xy"][1])],
        dx=float(solved["dx"]),
        total_mass=float(solved["info"]["total_mass"]),
        pose_kwargs={k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in solved["pose_kwargs"].items()},
        params={k: (list(v) if isinstance(v, tuple) else v)
                for k, v in dataclasses.asdict(params).items()},
    )
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    if verbose:
        print(f"[export] {path}")
        print(f"[export] {meta_path}")
    return meta


def load_meta(meta_path: str = SCENE_META) -> dict:
    import json

    with open(meta_path) as fh:
        return json.load(fh)


def load_scene(path: str = SCENE_XML) -> mujoco.MjModel:
    """Load the exported scene (no MjSpec needed; works on older MuJoCo)."""
    return mujoco.MjModel.from_xml_path(path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="build/inspect the brachiation scene")
    ap.add_argument("--export", default=None, metavar="XML",
                    help="export the scene to this path (writes the .json sidecar next "
                         "to it) instead of just printing the diagnostics")
    ap.add_argument("--spacing", type=float, default=None,
                    help="bar spacing in metres (default 0.40, the value every run so "
                         "far used).  0.35 is the M3.1 easy-geometry asset")
    ap.add_argument("--mjx-compat", type=int, default=0,
                    help="1 = capsule/primitive variant for the MJX-JAX backend, which "
                         "has a DIFFERENT keyframe and reach (see scene_bars*.json)")
    args = ap.parse_args()

    if args.export:
        kw = dict(mjx_compat=bool(args.mjx_compat))
        if args.spacing is not None:
            kw["spacing"] = args.spacing
        params = SceneParams(**kw)
        export_scene(params, path=args.export, verbose=False)
        print(f"exported {args.export} (spacing={params.spacing}, "
              f"mjx_compat={params.mjx_compat})")
        raise SystemExit(0)

    params = SceneParams()
    solved = solve_hang(params)
    info = solved["info"]
    print(f"total mass        : {info['total_mass']:.2f} kg")
    print(f"body weight       : {info['total_mass'] * 9.81:.1f} N")
    print(f"bar (x,z)         : ({solved['bar_xy'][0]:+.4f}, {solved['bar_xy'][1]:+.4f})")
    print(f"CoM               : {np.round(info['com'], 3)}")
    print(f"knuckle (L)       : {np.round(info['left']['knuckle'], 3)}")
    print(f"shoulder/elbow (L): {np.round(info['left']['shoulder'], 3)} {np.round(info['left']['elbow'], 3)}")
    print(f"CoM under bar dx  : {solved['dx']:+.4f} m")
    print(f"solved hip_pitch  : {solved['pose_kwargs']['hip_pitch']:+.3f}")
    model, _ = build_model(params, verbose=True)
    data = mujoco.MjData(model)
    reset_to(model, data)
    print("contacts at reset :", contact_report(model, data))
