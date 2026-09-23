"""Brachiation environment for JaxGCRL — plan A: physics via MJX, brax only as types.

Design (see ``docs/M2_任务变量定义.md`` v3 for the confirmed choices):

* **physics**: ``mujoco.mjx`` directly — ``mjx.put_model(impl=...)``,
  ``mjx.make_data``, ``jax.jit(mjx.step)``.  brax's ``PipelineEnv`` /
  ``brax.mjx.pipeline`` are **not** used; ``brax.envs.base.{Env,State}`` are used
  only as the (physics-free) interface JaxGCRL's CRL already relies on.
* **action** (14): 10 arm joints (shoulder p/r/y, elbow, wrist_roll per side),
  2 waist (yaw, pitch), 2 grasp scalars.  Legs are position-held at the keyframe;
  wrist pitch/yaw are equality-locked.
* **observation**: ``obs = [state | goal]``, ``state_dim`` = 141, goal = 10:
  ``[x_torso, z_torso, p_L(3), p_R(3), c_L, c_R]``.
* **control rate**: ``n_frames`` physics steps (dt=0.002) per env step; the
  confirmed default is 50 Hz (``n_frames=10``).
* **termination**: fall only (``root_z < bar_z - 0.8``); reaching the goal does
  not terminate.
* **contact budget** (MJX-Warp): ``naconmax`` is the contact capacity for **all
  worlds together**, so it must be scaled with the batch -- keep
  ``naconmax >= 128 * num_envs`` (e.g. 16384 for 128 envs).  If it is too small,
  Warp *silently drops contacts* (and prints a broadphase-overflow warning); check
  with ``mjx_backend.overflow_bits(data)``.

Registered as ``create_brachiation()``; JaxGCRL wraps it with
``TrajectoryIdWrapper`` + ``brax.envs.training.wrap`` unchanged.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

_SRC = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import mjx_backend as mb  # noqa: E402

# --------------------------------------------------------------------------- #
# task constants (kept in one place so the docs and the code cannot drift)
# --------------------------------------------------------------------------- #

ARM_WAIST_ACT = (  # order = the action vector's first 12 entries
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint",
)
# Action windows: radians of joint target per unit action (a in [-1, 1]) around the
# hang keyframe, in the order of ARM_WAIST_ACT.
#
#   "m1"    : the placeholder used for the first four runs.  It allows only +-0.2 rad
#             (+-11.5 deg) on every arm joint and +-0.25/0.15 on waist yaw/pitch.
#             MEASURED (src/m3_swing.py --action-box): the best |p_R - B1| this box can
#             even EXPRESS is 0.123 m, versus a 0.05 m grasp threshold and 0.031 m for a
#             free arm -> the policy could never command a reaching pose.  Kept only so
#             old runs stay reproducible.
#   "reach" : sized from the M3.0 reachability solve (needs ~0.46 rad shoulder pitch,
#             ~0.89 roll, ~1.61 elbow) plus margin; the waist gets ~87% of its pitch
#             range and +-57 deg of yaw.
ACTION_WINDOWS = {
    "m1": (0.20, 0.20, 0.20, 0.20, 0.20,
           0.20, 0.20, 0.20, 0.20, 0.20,
           0.25, 0.15),
    "reach": (0.70, 1.20, 0.80, 2.00, 0.70,
              0.70, 1.20, 0.80, 2.00, 0.70,
              1.00, 0.45),
}
ARM_WAIST_SCALE = ACTION_WINDOWS["m1"]   # default table for callers that import the name

FINGER_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")

TAU_GRASP = 0.05        # m, softmax temperature for "which bar"
TAU_CONTACT = 0.04      # m, soft contact indicator width
GRASP_THRESH = 0.05     # m, hand-to-bar distance counted as a grasp

# ---- real contact sensing (9.72) ------------------------------------------- #
# Every "is a hand on the bar" test above is a *geometric proxy* (distance from
# the calibrated grasp seat).  That is why a hand can hover 3 cm under the bar
# with the fingers wide open and still score like a grasp -- the exact pathology
# seen on goal=B1 (left hand parked near B1, never closing) and in the
# dual_hnext late regression (right hand 0.000 m from B1, grasp closure 0.12).
#
# The simulator does expose a real contact signal: ``data.qfrc_constraint`` (nv,)
# is the joint-space constraint load, i.e. J_contact^T @ efc_force, so summing
# |.| over one hand's 7 finger DOFs is the load that hand carries THROUGH ITS
# FINGERS.  This is also the cheapest honest sensor available here:
#   * the Warp backend exposes no ``data.contact`` / ``data.efc_force``;
#   * MJX implements only ``sensor_pos/vel/acc`` -- ``<touch>`` is not supported;
#   * ``qfrc_constraint`` IS in both ``jax`` and ``warp`` mjx.Data (shape (nv,)).
# Measured on the `hang` keyframe (d035, two-handed hang, fingers closed):
#   Sum|qfrc_constraint| over finger DOFs = 40.1 (L) / 41.0 (R) N m,
#   while the global max is 337.5 N = m*g (the bars carry the whole robot).
# With the fingers OPEN and the robot released: 0.00 / 0.00 (global max 0.10).
# Consequences for the design:
#   * a *distance* threshold is not a contact proxy at all -- binned by distance
#     to the nearest bar, the finger load is a median 62 N m even at 3.5-6 cm
#     (the fingers are loaded by the bar they are actually hooked on), so a
#     "hold" feature must threshold the FORCE, not the distance;
#   * the force alone cannot say WHICH bar a hand is loaded on, so the feature
#     below is the AND of "this hand is inside the goal bar's window" and "this
#     hand is carrying load" -- that is what removes the B0 loophole.
CONTACT_F0 = 40.0       # N m, reference load = one hand of a validated hang
                        # (static keyframe; through the control loop a real hang
                        # sits at 48-68 and a one-handed swinging hang bottoms at
                        # 37.7 -- measured, see src/check_contact_sense.py)
CONTACT_THETA = 0.3     # "loaded" cutoff as a fraction of F0 -> 12.0 N m.
                        # Measured band (src/check_contact_sense.py, sweep of
                        # 0.20/0.25/0.30/0.40/0.45/0.50 with 2x margins): the 18
                        # controls hold for theta in [0.25, 0.45] -> threshold
                        # 10-18 N m.  Below it a hand only BRUSHING the bar with
                        # open fingers (peak 9.5) counts as a grip; above it a
                        # one-handed swinging hang (min 37.7) loses its 2x margin.
                        # 0.3 is 1.3x above the brush and 3.1x below the swing,
                        # i.e. deliberately conservative on the "never call a
                        # hover a grip" side.
CONTACT_EMA = 5.0       # control steps, EMA time constant applied BEFORE the
                        # threshold (alpha = 1/CONTACT_EMA): single-step contact
                        # impulses (a graze, a soft-body bounce) must not latch a
                        # 25-step streak.

FALL_DEPTH = 0.8        # m below the bar plane -> terminated


class FeatureLayout(NamedTuple):
    root_pos: slice
    root_quat: slice
    root_linvel: slice
    root_angvel: slice
    joint_pos: slice
    joint_vel: slice
    hand_pos: slice
    grasp_dist: slice
    grasp_soft: slice
    grasp_ind: slice
    grasp_support: slice   # max(c_L, c_R): "at least one hand is on the bar"
    hold: slice            # sustained-contact feature 1-exp(-streak/HOLD_TAU)
    hold_dual: slice       # same, for "both hands on the SAME bar"
    cnext: slice           # [c_L,gk, c_R,gk]: per-hand contact with the GOAL bar
    hnext: slice           # [h_L,gk, h_R,gk]: per-hand SUSTAINED contact with it
    dnext: slice           # [d_L,gk, d_R,gk]: per-hand DISTANCE to it (metres)
    contact: slice         # [f_L,gk, f_R,gk]: per-hand SUSTAINED contact with the
                           # goal bar measured from the REAL contact load
                           # (qfrc_constraint), not from distance (see CONTACT_F0)
    lcontact: slice        # 9.74: BAR-AGNOSTIC load-gated trio
                           # [h_any_load, h_dual_load, h_both_load] -- see
                           # LCONTACT_FAMILY.  No k_goal dependence at all.
    park: slice            # how long the torso has been parked under some bar
    cross: slice           # [p_R - p_B1 (3), c_{R,B1}, c_{L,B0}]  (only variant "cross")
    prev_action: slice
    advance: slice         # M4: [dx, dz, c_L,next, c_R,next] relative to bar k_ref+1
    kref: slice            # M4: k_ref / (n_bars - 1)   (state only, NOT a goal entry)
    state_dim: int
    goal_indices: Tuple[int, ...]


GOAL_VARIANTS = ("full", "support", "support_hold", "support_dual", "dual_nomax",
                 "dual_cnext", "dual_hnext", "dual_hnext_max", "dual_hnext_dual",
                 "dual_hcontact", "support_dual_contact",
                 "support_dual_lc", "support_dual_lcb",
                 "dual6c", "dual6d", "park",
                 "position", "cross", "cross3", "hold2", "advance")

# ---- M4.0 "advance one bar" ------------------------------------------------
# The reference bar `k_ref` is the bar of the last *sustained* grip (>= K_SUSTAIN
# consecutive steps with a hand inside its grasp window).  Everything the `advance`
# goal asks for is expressed RELATIVE to bar k_ref+1, so the same goal means "the
# next bar" at every bar -- which is what makes multi-bar traversal representable
# with a single instruction goal.  Nothing here constrains the *strategy*: a
# ballistic dive is still allowed, but the goal only counts when the robot ends up
# hanging under the next bar with BOTH hands on it (the two contact entries are
# per-hand and target specific -- that is also what removes the one-handed
# attractor of `support`/`support_hold`, where max(c_L,c_R) was satisfied by one
# hand alone).
# A grip must last this long before it counts as having advanced k_ref.  It is the
# knob that couples the "arrive and STAY" pressure to the self-curriculum: the goal
# keeps pointing at the bar just grabbed until k_ref advances, so a longer gate
# means (a) more steps for `hold_next` to build (at 25 steps it reaches 0.92, i.e.
# `success` becomes reachable) and (b) the goal only moves on once the robot really
# is hanging there.  Set equal to the env's own dwell_steps (0.5 s).
# ---- M5 "park" goal (coarse, torso-only) ----------------------------------
# The whole terminal condition is "the torso is parked under the target bar and
# has been for a while".  Rationale + evidence: docs/M2_JaxGCRL接入记录.md 9.48.
# "parked under a bar" box for the coarse `park` goal.  Sizing follows the same
# logic as the 5 cm grasp threshold (see docs/M4 ... #8): too tight and a genuine
# stable hang flickers the streak, too loose and a swing-through counts as rest.
# Measured natural hang (CPU, d035, 25 settled steps): |dx| ~= 0.015-0.02,
# dz ~= -0.034 w.r.t. the keyframe torso height, so these are ~3x margins.
# x-coverage is 2*0.06/0.35 = 34% of the inter-bar gap, i.e. the box does not
# swallow the whole swing corridor.
PARK_RX = 0.06          # m, |x_torso - x_bar| for "parked under this bar"
PARK_RZ = 0.10          # m, |z_torso - z_hang|
K_SUSTAIN = 25


# "support" and "support_hold" share the max(c_L, c_R) state slot; "support_hold"
# adds one more goal dimension, the *sustained-contact* feature (see HOLD_TAU).
SUPPORT_FAMILY = ("support", "support_hold", "support_dual", "dual_nomax",
                  "dual_cnext", "dual_hnext",
                  # 9.66: these两个 = dual_hnext + 恰好一维（角色 2 / 角色 3）。
                  # 进 SUPPORT_FAMILY 只是为了拿到 max(c) 的 state 槽（dua_hnext_max
                  # 需要它作为 goal 项；dual_hnext_dual 只把它当可观测量）。
                  "dual_hnext_max", "dual_hnext_dual",
                  # 9.72: support_dual + 两个"真实接触"维（max(c) 仍作为角色 2）,
                  # 以及 dual_hnext 的 hnext(2) -> contact(2) 单变量替换版
                  "support_dual_contact", "dual_hcontact",
                  # 9.74: bar-agnostic load trio = support_dual + 2-3 dims
                  "support_dual_lc", "support_dual_lcb")
# "dual6c"/"dual6d" are deliberately NOT in SUPPORT_FAMILY: their goal drops the
# max(c)/h_any/h_dual block entirely (x,z + goal-bar per-hand terms only).
# "support_hold" adds the *any-hand* sustained-contact entry h_any; "support_dual"
# adds a second one, h_dual, for "both hands inside the SAME bar's window" (see
# docs/M4_对与#11的分析和goal的设计问题的讨论.md 5).  The two together give the goal
# space an explicit three-layer structure -- flight (0,0) -> single support (1,0)
# -> dual support (1,1) -- without constraining how the policy moves between them.
HOLD_FAMILY = ("support_hold", "support_dual", "dual_nomax", "dual_cnext",
               "dual_hnext", "support_dual_contact",
               # 9.72: dual_hcontact carries h_any/h_dual as OBSERVABLES only, so
               # that its state is dual_hnext's state with hnext(2) -> contact(2)
               # (a clean single-variable probe, not a state-size change too)
               "dual_hcontact", "support_dual_lc", "support_dual_lcb")
# "dual_nomax" = run #13's support_dual minus the instantaneous `max(c_L,c_R)`
# entry (goal = [x, z, p_L(3), p_R(3), h_any, h_dual], 10-D).  The rationale above
# was FALSIFIED by run #18 + CPU measurement (docs/M2_JaxGCRL接入记录.md 9.50):
# that entry is not a saturating constant -- a settled hang reads 0.52 against a
# goal of 1.0, the trained policy drives it to 0.96-0.98, and deleting it makes
# training collapse (fell 100%, advance/succ stuck at 0).  Kept for the record.
DUAL_FAMILY = ("support_dual", "dual_nomax", "dual_cnext", "dual_hnext",
               "dual_hnext_dual", "support_dual_contact", "dual_hcontact",
               "support_dual_lc", "support_dual_lcb")
# "dual_cnext" (9.51) keeps a high-gain instantaneous contact term -- the measured
# load-bearing part -- but makes it PER-HAND and aimed at the INSTRUCTION GOAL BAR:
#   g = [x, z, p_L(3), p_R(3), c_L,gk, c_R,gk, h_any, h_dual]      (12-D)
# instead of max(c_L,c_R) = "at least one hand on SOME bar", which the start-bar
# hang already satisfies and a single hand can satisfy alone.
CNEXT_FAMILY = ("dual_cnext", "dual_hnext", "dual6c",
                "dual_hnext_max", "dual_hnext_dual", "dual_hcontact")
# "dual_hnext" (9.52) = dual_cnext with the two bar-AGNOSTIC persistence dims
# h_any/h_dual replaced by per-hand, goal-bar-specific ones:
#   g = [x, z, p_L(3), p_R(3), c_L,gk, c_R,gk, h_L,gk, h_R,gk]      (12-D)
# h_any is satisfied by the *start* bar (0.92 while parked on B0 with the goal on
# B1) and h_dual by any single bar, so both are "already 92% done" at t=0; they
# also reward simply waiting.  h_h,gk only rises while THAT hand stays in the GOAL
# bar's window -- no loophole, and the passive "wait for h to grow" shortcut is
# gone.  h_any/h_dual stay in the STATE as observable context (and metrics), just
# not in the goal.
HPAIR_FAMILY = ("dual_hnext", "dual6c", "dual6d",
                "dual_hnext_max", "dual_hnext_dual")
# "dual_hnext_max"  = dual_hnext + max(c_L,c_R)（任意杆的抓握质量，§9.66 角色 2）
# "dual_hnext_dual" = dual_hnext + h_dual（同杆双手持续，§9.66 角色 3）
# 两者都只比 dual_hnext 多 1 维，见 9.66 的分工表。
HMAX_FAMILY = ("dual_hnext_max",)
HDMAX_FAMILY = ("dual_hnext_dual",)
# "dual6c" / "dual6d" (9.54): the reduced 6-D goal -- torso (x,z) + per-hand
# contact/distance with the goal bar + per-hand sustained contact:
#   dual6c = [x, z, c_L,gk, c_R,gk, h_L,gk, h_R,gk]
#   dual6d = [x, z, d_L,gk, d_R,gk, h_L,gk, h_R,gk]      (metres, linear)
# p_L/p_R (6 world coordinates) leave the GOAL but stay in the STATE: the user's
# point is that the per-hand goal-bar contact already says "hand on the goal bar",
# at a far higher gain, so 6 more absolute coordinates are redundant in the goal.
# The two differ ONLY in the contact encoding, which is the one-variable probe for
# the 9.50/9.51 mechanism question: c = exp(-(d/0.04)^2) is ~0 with ~0 gradient
# beyond ~10 cm (no reach direction), while d is linear.  In the SATURATED regime
# d gives the approach direction the plan asks for ("per-hand distance, not the
# saturating soft contact" -- see _advance_features), and c gives the grip magnitude
# that run #18 showed to be load-bearing near the bar.
DNEXT_FAMILY = ("dual6d",)
# ---- 9.72: the "real contact" family --------------------------------------- #
# Both variants carry two extra state/goal dims, f_{h,gk} for h = L, R:
#
#   f_{h,gk} = 1 - exp(-s_h / HOLD_TAU),
#   s_h      = consecutive steps with  (d[h, gk] < GRASP_THRESH) AND (F_h > thresh)
#
# where F_h is the EMA (alpha = 1/CONTACT_EMA) of Sum|qfrc_constraint| over hand
# h's 7 finger DOFs and thresh = CONTACT_F0 * CONTACT_THETA.  The AND is what
# makes it honest: the force says "this hand really is loaded", the distance gate
# says "and it is loaded on the INSTRUCTED bar" (a hand clamped on the start bar
# also carries 40 N m, so the force alone has the B0 loophole).
#
# The two variants answer the two open failures with ONE change each:
#   dual_hcontact        = dual_hnext with h_L,gk/h_R,gk -> f_L,gk/f_R,gk
#                          (same 12-D goal, same state, single-variable probe for
#                          "the trailing hand reaches B1 at 0.000 m but never
#                          closes" -- hnext's distance criterion is satisfied by
#                          hovering, f is not);
#   support_dual_contact = support_dual + the same two dims (11 -> 13 dims), i.e.
#                          the two-bar-curriculum recipe with "load the instructed
#                          bar per hand" added on top of max(c)/h_any/h_dual.
# h_any/h_dual themselves are NOT re-encoded: max(f_L,f_R) and min(f_L,f_R) are
# exactly the bar-agnostic "any hand / both hands loaded on the goal bar", so the
# two bar-specific dims already contain them (with the extra bar identity).
CONTACT_FAMILY = ("dual_hcontact", "support_dual_contact")

# ---- 9.74: the BAR-AGNOSTIC load family ------------------------------------ #
# `support_dual_contact` proved that a *bar-specific* load term (f_{h,gk}) removes
# the hovering pathology on the instructed bar, and also proved the cost: on the
# goal=B2 probe the robot dives straight at B2 and NEVER touches B1 before it
# (0 steps on B1 vs 17-19 for the bar-agnostic `support_dual` baseline), 56% of
# its live phase airborne.  The goal said "be loaded on B2"; B1 earned nothing.
#
# The baseline's B2 failure is a *load* failure, measured directly (goal-bar 2,
# 50 M steps, live phase, left hand at B2): it is inside B2's 5 cm window for
# 25-33% of the steps but carries load there for only 0-15%, with 0-9 steps of
# pure hover -- i.e. the distance criteria (`c` = 0.47-0.82, "in contact") do not
# make the hand actually grip.
#
# So the two fixes are orthogonal and both are needed:
#   * BAR-AGNOSTIC (this family): credit any bar, so the B1 stepping stone keeps
#     paying -- no dive, no dependence on the sampled instruction bar;
#   * LOAD-GATED: "in a window" alone is not a grip (the CONTACT_F0 threshold).
# With the per-hand boolean
#     g_h = (d[h, argmin_k d[h,k]] < GRASP_THRESH)  AND  (EMA_5(F_h) > F0*theta)
# (i.e. "this hand is really loaded, on whichever bar it is on"), the trio is
#
#   h_any_load  : 1-exp(-S/10), S = steps with ANY hand's g_h
#   h_dual_load : same, with BOTH hands loaded AND inside the SAME bar's window
#   h_both_load : same, with BOTH hands loaded (on whatever bars)
#
# `h_any_load` is the load version of `h_any`; `h_dual_load` of `h_dual`.  They
# are ADDED to `support_dual` rather than replacing anything: run #18 showed that
# deleting the continuous grip term collapses training (9.50), and the distance
# versions are what let the chain form in the first place.  `h_both_load` is the
# one term that targets the REACHING hand during a hand-over -- `h_any_load` is
# already satisfied by the supporting hand, and `h_dual_load` says nothing about
# which hand, so neither can demand that the hand that just arrived takes load.
LCONTACT_FAMILY = ("support_dual_lc", "support_dual_lcb")
# `hold = 1 - exp(-streak / HOLD_TAU)` where `streak` is the number of consecutive
# steps with at least one hand inside the grasp window of some bar (< GRASP_THRESH,
# the same criterion as bar_L/bar_R/max_bar -- NOT max(c) > 0.5, which would call a
# settled hang "not gripping"; see docs/M2_JaxGCRL接入记录.md 9.34.2).  HOLD_TAU =
# 10 steps = 0.2 s: a smooth, dense stand-in for "hung on and stayed" -- 5 steps ->
# 0.39, 10 -> 0.63, 25 (dwell_steps) -> 0.92.  Deliberately NOT a hard time
# threshold: run #10 only ever held B1 for 4-7 steps, so a 25-step indicator would
# give no gradient at all.
HOLD_TAU = 10.0

# The "cross family" of goals (M3 first crossing).  They share one state block
#   [rel_pR(3), d_RB1, c_RB1, c_LB0, c_LB1, c_RB0]
# where rel_pR = p_R - p_B1, d_RB1 = |rel_pR| (x,z), and c_{h,k} is the soft contact
# of hand h with bar k.  The variants pick from it:
#   "cross"  : [rel_pR(3), c_RB1, c_LB0]        -- swing hand AT the target bar AND
#               the support hand STILL on the original bar (a fly-through cannot
#               satisfy it: letting go of B0 drives c_LB0 -> 0)
#   "cross3" : [d_RB1, c_RB1, c_LB0]            -- same task, but the smooth term is
#               the scalar distance instead of a hand pose (no absolute coordinates,
#               no redundancy with the saturating contact term)
#   "hold2"  : [c_LB1, c_RB1]                   -- pure contacts: "stably stopped on
#               the TARGET bar with both hands".  Coarser goal space (2 bits), so the
#               relabeled-goal curriculum has little granularity -- use it for the
#               end state, and note a momentary fly-through can satisfy it.
CROSS_TARGET = 1      # target bar for the swing (right) hand
CROSS_SUPPORT = 0     # bar the support (left) hand must keep
CROSS_FAMILY = ("cross", "cross3", "hold2")


def default_layout(njoints: int, goal_variant: str = "full") -> FeatureLayout:
    """Observation layout + which entries form the goal.

    ``goal_variant`` decides how "being on the bar" enters the goal:

    * ``full``     : ``[x, z, p_L(3), p_R(3), c_L, c_R]`` (10-D).  Requires BOTH
      hands to grasp, so "let go of one hand" is *farther* from a hang goal than
      staying put -- exactly the intermediate a brachiation step must pass through
      (``docs/M2_JaxGCRL接入记录.md`` §9.22).
    * ``support``  : ``[x, z, p_L(3), p_R(3), max(c_L,c_R)]`` (9-D).  Releasing ONE
      hand is free (the other still supports), but losing the last grip is
      penalised -- the semantics brachiation actually needs (see §9.25: with
      ``position`` alone the policy learned to hang by hooking the wrist with both
      fingers open, which is both invisible to the goal and eventually unstable).
    * ``position`` : ``[x, z, p_L(3), p_R(3)]`` (8-D), grasp ignored entirely.

    The state keeps ``c_L, c_R`` (and now ``max(c_L,c_R)``) in every variant, so
    the policy always observes the contact; only the goal definition changes.
    """
    if goal_variant not in GOAL_VARIANTS:
        raise ValueError(f"goal_variant must be one of {GOAL_VARIANTS}, got {goal_variant!r}")
    i = 0
    def nxt(k):
        nonlocal i
        s = slice(i, i + k)
        i += k
        return s
    root_pos = nxt(3)
    root_quat = nxt(4)
    root_linvel = nxt(3)
    root_angvel = nxt(3)
    joint_pos = nxt(njoints)
    joint_vel = nxt(njoints)
    hand_pos = nxt(6)        # p_L, p_R
    grasp_dist = nxt(10)     # d[hand, bar]
    grasp_soft = nxt(10)     # softmax(-d/tau)
    grasp_ind = nxt(2)       # soft contact indicator c_L, c_R
    # max(c_L, c_R) is only a *goal* entry, so it only takes a state slot in the
    # "support" variants -- that keeps "full"/"position" observation sizes (and
    # therefore previously saved checkpoints) unchanged.
    grasp_support = nxt(1) if goal_variant in SUPPORT_FAMILY else slice(i, i)
    # "support_hold" additionally carries how *long* a hand has been on a bar.
    # It comes from `info["hold_streak"]` (it is history, not a function of the
    # instantaneous state), which is why `_state_features`/`_achieved_goal` take
    # it as an extra argument.
    hold = nxt(1) if goal_variant in HOLD_FAMILY else slice(i, i)
    hold_dual = nxt(1) if goal_variant in DUAL_FAMILY else slice(i, i)
    # per-hand contact with the GOAL bar (2 dims, "dual_cnext" only)
    cnext = nxt(2) if goal_variant in CNEXT_FAMILY else slice(i, i)
    hnext = nxt(2) if goal_variant in HPAIR_FAMILY else slice(i, i)
    dnext = nxt(2) if goal_variant in DNEXT_FAMILY else slice(i, i)
    # per-hand sustained contact with the GOAL bar measured from the REAL contact
    # load (9.72).  Only allocated for the contact family, so observation sizes --
    # and therefore previously saved checkpoints -- are untouched for every other
    # variant.
    contact = nxt(2) if goal_variant in CONTACT_FAMILY else slice(i, i)
    # 9.74 bar-agnostic load trio (no k_goal dependence)
    lcontact = nxt(3) if goal_variant in LCONTACT_FAMILY else slice(i, i)
    # "park": how long the TORSO has been inside some bar's hang box (history)
    park = nxt(1) if goal_variant == "park" else slice(i, i)
    # cross-family extras: [rel_pR(3), d_RB1, c_RB1, c_LB0, c_LB1, c_RB0]
    cross = nxt(8) if goal_variant in CROSS_FAMILY else slice(i, i)
    prev_action = nxt(14)
    # M4 "advance" block: its six dims are goal entries AND therefore must live in
    # the state, plus k_ref (context for the critic, NOT a goal entry).
    advance = nxt(6) if goal_variant == "advance" else slice(i, i)
    kref = nxt(1) if goal_variant == "advance" else slice(i, i)
    goal_indices = (root_pos.start, root_pos.start + 2,
                    *(range(hand_pos.start, hand_pos.start + 6)))
    if goal_variant == "dual_hcontact":
        # [x, z, p(6), c_L,gk, c_R,gk, f_L,gk, f_R,gk]  (12)
        # dual_hnext's goal with the two SUSTAINED dims re-measured from contact
        # load instead of distance (single-variable swap, see CONTACT_FAMILY).
        goal_indices = goal_indices + (cnext.start, cnext.start + 1,
                                       contact.start, contact.start + 1)
    elif goal_variant in LCONTACT_FAMILY:
        # [x, z, p(6), max(c), h_any, h_dual] + [h_any_load, h_dual_load(, h_both_load)]
        base = (grasp_support.start, hold.start, hold_dual.start,
                lcontact.start, lcontact.start + 1)
        goal_indices = goal_indices + (base if goal_variant == "support_dual_lc"
                                       else base + (lcontact.start + 2,))
    elif goal_variant == "support_dual_contact":
        # [x, z, p(6), max(c), h_any, h_dual, f_L,gk, f_R,gk]  (13)
        goal_indices = goal_indices + (grasp_support.start, hold.start,
                                       hold_dual.start,
                                       contact.start, contact.start + 1)
    elif goal_variant in HMAX_FAMILY:
        # [x, z, p(6), max(c), c_L,gk, c_R,gk, h_L,gk, h_R,gk]   (13)
        goal_indices = goal_indices + (grasp_support.start,
                                       cnext.start, cnext.start + 1,
                                       hnext.start, hnext.start + 1)
    elif goal_variant in HDMAX_FAMILY:
        # [x, z, p(6), c_L,gk, c_R,gk, h_L,gk, h_R,gk, h_dual]   (13)
        goal_indices = goal_indices + (cnext.start, cnext.start + 1,
                                       hnext.start, hnext.start + 1,
                                       hold_dual.start)
    elif goal_variant in ("dual6c", "dual6d"):
        # [x, z, <goal-bar per-hand term>(2), h_L,gk, h_R,gk] -- note: NOT `_pos`,
        # i.e. p_L/p_R are context-only for these variants.
        terms = cnext if goal_variant == "dual6c" else dnext
        goal_indices = (root_pos.start, root_pos.start + 2,
                        terms.start, terms.start + 1,
                        hnext.start, hnext.start + 1)
    elif goal_variant == "full":
        goal_indices = goal_indices + (grasp_ind.start, grasp_ind.start + 1)
    elif goal_variant in SUPPORT_FAMILY:
        # "dual_nomax" deliberately does NOT put max(c_L,c_R) in the goal (it stays
        # in the state as observable context); every other support variant does.
        if goal_variant != "dual_nomax" and goal_variant not in CNEXT_FAMILY:
            goal_indices = goal_indices + (grasp_support.start,)
        if goal_variant in CNEXT_FAMILY:
            # order: [x, z, p(6), c_L,gk, c_R,gk, ...]
            goal_indices = goal_indices + (cnext.start, cnext.start + 1)
        if goal_variant in HPAIR_FAMILY:
            # "dual_hnext": per-hand sustained contact with the GOAL bar replaces
            # the bar-agnostic h_any/h_dual in the goal (they stay in the state).
            goal_indices = goal_indices + (hnext.start, hnext.start + 1)
        else:
            if goal_variant in HOLD_FAMILY:
                goal_indices = goal_indices + (hold.start,)
            if goal_variant in DUAL_FAMILY:
                goal_indices = goal_indices + (hold_dual.start,)
    elif goal_variant in CROSS_FAMILY:
        # indices INTO the state block: [0..2]=rel_pR, 3=d_RB1, 4=c_RB1, 5=c_LB0,
        # 6=c_LB1, 7=c_RB0
        pick = {"cross": (0, 1, 2, 4, 5), "cross3": (3, 4, 5), "hold2": (6, 4)}[goal_variant]
        goal_indices = tuple(cross.start + k for k in pick)
    elif goal_variant == "park":
        # g = [x_torso, z_torso, h_park]: absolute torso coordinates (which carry
        # the direction AND the bar identity) + "and it has been parked there"
        goal_indices = (root_pos.start, root_pos.start + 2, park.start)
    elif goal_variant == "advance":
        # goal = [dx, dz, d_L,next, d_R,next, hold_next, progress]
        # (k_ref stays context: it is progress, not a target)
        goal_indices = tuple(advance.start + k for k in range(6))
    return FeatureLayout(root_pos, root_quat, root_linvel, root_angvel, joint_pos,
                         joint_vel, hand_pos, grasp_dist, grasp_soft, grasp_ind,
                         grasp_support, hold, hold_dual, cnext, hnext, dnext, contact,
                         lcontact, park, cross, prev_action,
                         advance, kref, i, tuple(goal_indices))


class Brachiation(Env):
    """Monkey-bar (brachiation) environment on the Unitree G1."""

    def __init__(
        self,
        scene_xml: str,
        impl: str = "warp",
        n_frames: int = 10,
        n_bars: int = 5,
        reset_noise_root: float = 0.005,
        reset_noise_arm: float = 0.01,
        reset_noise_leg: float = 0.02,
        reset_noise_vel: float = 0.02,
        goal_bar: Optional[int] = None,      # None -> sample per episode
        goal_bar_min: int = 0,               # sampling range is [goal_bar_min, goal_bar_max]
        goal_bar_max: int = -1,              # -1 -> n_bars-1.  With start_bar_max > 0 the
                                             # sampled goal is kept STRICTLY AHEAD of the
                                             # start bar (see the curriculum in 9.68).
        goal_variant: str = "full",          # "full" | "support" | "position" (see default_layout)
        action_window: str = "reach",        # "m1" (old, cannot express a reach) | "reach"
        goal_reach_thresh: float = 0.35,
        contact_f0: float = CONTACT_F0,       # 9.72: reference finger load (N m)
        contact_theta: float = CONTACT_THETA, # 9.72: loaded cutoff / F0
        contact_ema: float = CONTACT_EMA,     # 9.72: EMA steps before thresholding
        dwell_steps: int = 25,               # 0.5 s at 50 Hz
        start_bar_max: int = 0,              # M4: reset on bar U{0..start_bar_max}
        goal_ahead: bool = False,            # M4: sample the goal bar AFTER the start bar
        goal_ahead_max: int = 0,             # M4: cap that distance (0 = no cap)
        naconmax: int = 16384,               # contact budget for ALL worlds (see note)
        njmax: int = 512,                    # constraint rows per world
        ls_iterations: Optional[int] = None, # None -> keep the scene XML value (20)
    ):
        if impl == "warp":
            mb.enable_warp_compat()
        self.impl = impl
        self.n_frames = int(n_frames)
        self.scene_xml = scene_xml
        self.mj_model = mujoco.MjModel.from_xml_path(scene_xml)
        # The exported scene pins ls_iterations=20 (vs MuJoCo's default 50) for
        # speed; M1 was validated with that value.  Allow overriding it so the
        # solver-convergence question can be A/B tested (see check_reset --ls-iterations).
        if ls_iterations is not None:
            self.mj_model.opt.ls_iterations = int(ls_iterations)
        self.ls_iterations = int(self.mj_model.opt.ls_iterations)
        self.mx_model = mb.silence_warp_overflow(mjx.put_model(self.mj_model, impl=impl), impl)
        self._data0 = mjx.make_data(
            self.mj_model, impl=impl,
            **({"naconmax": naconmax, "njmax": njmax} if impl == "warp" else {}))

        m = self.mj_model
        self.njoints = m.nq - 7
        self.goal_variant = str(goal_variant)
        self.layout = default_layout(self.njoints, self.goal_variant)
        self.goal_position_only = self.goal_variant == "position"   # kept for older callers
        self.state_dim = self.layout.state_dim
        # NOTE: keep this a plain tuple of Python ints.  JaxGCRL does
        # ``tuple(train_env.goal_indices)`` and passes it as a *static* jit
        # argument (``flatten_batch``); a jnp array there turns into tracers and
        # raises "Non-hashable static arguments are not supported".
        self.goal_indices = tuple(int(i) for i in self.layout.goal_indices)
        # ``_achieved_goal`` builds the full [x,z,p(6),c(2)] and then keeps these
        # entries, so the achieved goal and the goal set can never disagree.
        _pos = [0, 1] + [2 + j for j in range(6)]        # x, z, p_L, p_R
        # superset built by _features_numpy/_achieved_goal:
        #   [x, z, p_L(3), p_R(3), c_L, c_R, c_support, rel_pR(3), c_R_B1, c_L_B0]
        # superset = [x, z, p_L(3), p_R(3), c_L, c_R, c_sup] + cross block (8)
        _cross0 = 11
        self._goal_full_idx = jnp.array(
            {"full": _pos + [8, 9],
             "support": _pos + [10],
             # support + sustained contact (`hold`, appended last to the superset)
             "support_hold": _pos + [10, 19],
             # support + h_any + h_dual (both appended after the superset's hold)
             "support_dual": _pos + [10, 19, 20],
             # ... and support_dual WITHOUT the instantaneous max(c_L,c_R) ("c_sup",
             # superset index 10): what run #14 should have been -- run #13's recipe
             # minus the one saturating, direction-free goal entry.
             "dual_nomax": _pos + [19, 20],
             # per-hand contact with the goal bar (superset indices 28, 29)
             "dual_cnext": _pos + [28, 29, 19, 20],
             # per-hand sustained contact with the goal bar (superset 28..31)
             "dual_hnext": _pos + [28, 29, 30, 31],
             # reduced 6-D goals: NO hand positions, only x,z + the goal-bar terms
             # (superset 28,29 = c_*,gk; 30,31 = h_*,gk; 32,33 = d_*,gk)
             # 9.66: dual_hnext 各加一维（10 = max(c)，20 = h_dual）
             "dual_hnext_max": _pos + [10, 28, 29, 30, 31],
             "dual_hnext_dual": _pos + [28, 29, 30, 31, 20],
             # 9.72: real-contact sustained dims (superset 34, 35 = f_L,gk, f_R,gk)
             "dual_hcontact": _pos + [28, 29, 34, 35],
             "support_dual_contact": _pos + [10, 19, 20, 34, 35],
             # 9.74 bar-agnostic load trio (superset 36, 37, 38)
             "support_dual_lc": _pos + [10, 19, 20, 36, 37],
             "support_dual_lcb": _pos + [10, 19, 20, 36, 37, 38],
             "dual6c": [0, 1, 28, 29, 30, 31],
             "dual6d": [0, 1, 32, 33, 30, 31],
             # M5 coarse goal: absolute torso (x,z) + the parked streak (index 27)
             "park": [0, 1, 27],
             "position": list(_pos),
             "cross": [11, 12, 13, 15, 16],
             "cross3": [14, 15, 16],
             "hold2": [17, 15],
             # M4 superset tail: [.., hold(19), dx, dz, d_Lnext, d_Rnext,
             #                    hold_next, progress]
             "advance": [21, 22, 23, 24, 25, 26]}[self.goal_variant],
            dtype=jnp.int32)
        self.goal_size = int(len(self.goal_indices))
        self.goal_reach_thresh = goal_reach_thresh
        # ---- 9.72 contact-load thresholds -----------------------------------
        # Validated positively AND negatively (src/check_contact_sense.py): a
        # settled keyframe hang must read "loaded" on both hands for >=95% of the
        # steps, and a released robot (fingers open) for <=1%.
        self.contact_f0 = float(contact_f0)
        self.contact_theta = float(contact_theta)
        self.contact_ema = float(contact_ema)
        if self.contact_f0 <= 0 or self.contact_theta <= 0:
            raise ValueError(f"contact_f0/contact_theta must be > 0, got "
                             f"{self.contact_f0}/{self.contact_theta}")
        if self.contact_ema < 1.0:
            raise ValueError(f"contact_ema must be >= 1 step, got {self.contact_ema}")
        self.contact_thresh = self.contact_f0 * self.contact_theta
        self.dwell_steps = int(dwell_steps)
        self.n_bars = int(n_bars)
        self._reset_cfg = (reset_noise_root, reset_noise_arm, reset_noise_leg, reset_noise_vel)

        # --- indices ---------------------------------------------------------
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: int(np.flatnonzero(m.actuator_trnid[:, 0] == jid(n))[0])
        self.act_armwaist = jnp.array([aid(n) for n in ARM_WAIST_ACT], dtype=jnp.int32)
        if action_window not in ACTION_WINDOWS:
            raise ValueError(f"action_window must be one of {list(ACTION_WINDOWS)}")
        self.action_window = action_window
        self.act_scale = jnp.array(ACTION_WINDOWS[action_window], dtype=jnp.float32)
        self.hand_act = {s: jnp.array([aid(f"{s}_hand_{j}_joint") for j in FINGER_JOINTS], dtype=jnp.int32)
                         for s in ("left", "right")}
        # DOF addresses (into qvel / qfrc_constraint, i.e. nv) of the same finger
        # joints -- the contact-load readout of 9.72.  NB: `jnt_dofadr`, NOT the
        # actuator index: nv = 49 while there are 43 actuators (the 6-DoF free
        # joint occupies 0..5), so the two differ by 6 for every joint.
        self.finger_dof = {
            s: jnp.array([int(m.jnt_dofadr[jid(f"{s}_hand_{j}_joint")])
                          for j in FINGER_JOINTS], dtype=jnp.int32)
            for s in ("left", "right")}
        self.wrist_body = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_roll_link")
                           for s in ("left", "right")}
        # --- keyframe --------------------------------------------------------
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "hang")
        kd = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, kd, kid)
        mujoco.mj_forward(m, kd)
        self.key_qpos = jnp.array(kd.qpos.copy())
        self.key_ctrl = jnp.array(kd.ctrl.copy())

        # bar geometry (bars are welded to the world, axis = +Y)
        bpos = np.array([m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"bar{i}")]
                         for i in range(self.n_bars)])
        self.bar_x = jnp.array(bpos[:, 0], dtype=jnp.float32)
        # Torso x at the settled hang under each bar.  The body does NOT hang
        # directly under the bar it holds: the hands hook over the bar, so the
        # keyframe torso x is offset from its bar by `dx_goal` (same quantity
        # `_features_numpy`/`_advance_features` already use).  Measured on d035:
        # bar_x[0] = 0.056 while the hanging torso sits at -0.015, i.e. dx_goal
        # ~= -0.071 m.  Anything that asks "is the torso hanging under bar k"
        # must compare against THIS, not against bar_x -- centring the box on
        # bar_x puts the natural hang right on the boundary, which makes the
        # sustained-contact streaks flicker (exactly the failure mode the M4 doc
        # section 8 warns about for the 5 cm grasp threshold).
        self.hang_x = self.bar_x + (float(kd.qpos[0]) - float(self.bar_x[0]))
        self.bar_z = float(bpos[0, 2])
        self.bar_spacing = float(bpos[1, 0] - bpos[0, 0]) if self.n_bars > 1 else 0.0

        # --- the bar seat expressed in each wrist frame (constant) -----------
        self.seat_local = {}
        for s in ("left", "right"):
            bid = self.wrist_body[s]
            R = kd.xmat[bid].reshape(3, 3)
            self.seat_local[s] = jnp.array(R.T @ (bpos[0] - kd.xpos[bid]), dtype=jnp.float32)

        # --- goal set: the canonical hang at each bar ------------------------
        # built as the full [x,z,p(6),c(2)] and then subset, so the 8-D and 10-D
        # variants share one construction path
        goal0 = self._features_numpy(kd, np.zeros(14, dtype=np.float32))
        # [x, z, p_L(3), p_R(3), c_L, c_R, c_support, rel_pR(3), c_R_B1, c_L_B0]:
        # only x and the hand x's shift; the cross entries are already relative
        # to a bar, so they do not move with k.
        shift = np.array([self.bar_spacing, 0.0] + [self.bar_spacing, 0, 0] * 2
                         + [0.0, 0.0, 0.0]                     # c_L, c_R, c_support
                         + [0.0] * 8                           # cross block (all relative)
                         + [0.0, 0.0]                          # hold, hold_dual (settled)
                         + [0.0] * 6                           # advance block (all relative)
                         + [0.0,                               # park (settled = 1)
                            0.0, 0.0,                          # c_L,next / c_R,next = 1
                            0.0, 0.0,                          # h_L,next / h_R,next = 1
                            0.0, 0.0,                          # d_L,next / d_R,next = 0
                            0.0, 0.0,                          # f_L,next / f_R,next = 1
                            0.0, 0.0, 0.0],                    # load trio = 1 (settled)
                         dtype=np.float32)
        keep = np.asarray(self._goal_full_idx)
        if self.goal_variant in CROSS_FAMILY:
            # the goal is a *task state*, not one of the canonical hang poses:
            self.goal_set = jnp.array([{
                # swing hand at B1 (rel 0) + gripping B1 + support still on B0
                "cross": [0.0, 0.0, 0.0, 1.0, 1.0],
                # same task, smooth term = distance to B1
                "cross3": [0.0, 1.0, 1.0],
                # stably stopped on B1 with both hands
                "hold2": [1.0, 1.0],
            }[self.goal_variant]], dtype=jnp.float32)
        elif self.goal_variant == "advance":
            # ONE bar-independent goal: "hanging under the next bar with both hands,
            # relaxed".  [dx, dz, d_Lnext, d_Rnext, hold_next, progress] -- relative,
            # same vector for every bar; `keep` selects exactly those entries.
            self.goal_set = jnp.array([goal0[keep]], dtype=jnp.float32)
        else:
            self.goal_set = jnp.array(
                np.stack([goal0 + k * shift for k in range(self.n_bars)])[:, keep])
        self.goal_bar = goal_bar
        self.goal_bar_min = int(goal_bar_min)
        self.goal_bar_max = (self.n_bars - 1 if int(goal_bar_max) < 0
                             else int(goal_bar_max))
        # Starting-bar randomisation is only meaningful for the bar-independent
        # "advance" goal: every other variant's goal_set is pinned to absolute bar
        # coordinates, so shifting the start would make it unreachable.
        self.start_bar_max = int(start_bar_max)
        # `start_bar_max` shifts the whole robot by k0 bars.  That is fine for any
        # variant whose goal_set covers the bars (the goal coordinates for bar gk are
        # absolute and correct whatever bar the episode starts on); what must not
        # happen is a goal bar *behind* the start, which is why `goal_ahead` exists.
        self.goal_ahead = bool(goal_ahead)
        self.goal_ahead_max = int(goal_ahead_max)
        if self.goal_ahead and self.goal_bar is not None:
            raise ValueError("goal_ahead requires a sampled goal bar "
                             "(train.py: --train-goal-bar -1)")
        if self.goal_bar_max < self.goal_bar_min:
            raise ValueError(f"goal_bar_max ({self.goal_bar_max}) < "
                             f"goal_bar_min ({self.goal_bar_min})")
        if (self.goal_bar is None and self.start_bar_max > 0
                and self.goal_bar_max < self.start_bar_max + 1):
            # every k0 in {0..start_bar_max} must have at least one bar ahead of it
            raise ValueError(
                f"with start_bar_max={self.start_bar_max} the sampled goal must be "
                f"strictly ahead of the start, so goal_bar_max must be >= "
                f"{self.start_bar_max + 1}; got {self.goal_bar_max}")
        # ---- curriculum table (9.68.1) --------------------------------------
        # With a randomised start we sample the DISTANCE first and the start second:
        #   * delta ~ U over the feasible distances,
        #   * k0    ~ U over the starts that admit that delta.
        # This is what makes the mixture sensible: for the two-bar curriculum
        # (start<=1, goal in [1,2]) delta=1 admits k0 in {0,1} and delta=2 only k0=0,
        # so the episode types come out as B0->B2 50% / B0->B1 25% / B1->B2 25%.
        # (The previous k0-first scheme gave B1->B2 50%, which over-weights the
        # *second* one-bar copy and under-weights the chaining case.)
        self._delta, self._klo, self._khi = [], [], []
        if self.goal_bar is None and self.start_bar_max > 0:
            _dmax = self.goal_bar_max
            if self.goal_ahead and self.goal_ahead_max > 0:
                _dmax = min(_dmax, self.goal_ahead_max)
            for _d in range(1, _dmax + 1):
                _lo = max(0, self.goal_bar_min - _d)
                _hi = min(self.start_bar_max, self.goal_bar_max - _d)
                if _lo <= _hi:
                    self._delta.append(_d); self._klo.append(_lo); self._khi.append(_hi)
            if not self._delta:
                raise ValueError(
                    f"no feasible (start, goal) pair for start_bar_max={self.start_bar_max}, "
                    f"goal_bar_min={self.goal_bar_min}, goal_bar_max={self.goal_bar_max}")
            self._delta = jnp.asarray(self._delta, jnp.int32)
            self._klo = jnp.asarray(self._klo, jnp.int32)
            self._khi = jnp.asarray(self._khi, jnp.int32)
        if self.goal_ahead and self.start_bar_max > self.n_bars - 2:
            # starting on the last bar leaves no bar ahead -> the goal would equal
            # the start (a trivially satisfied episode)
            raise ValueError(f"goal_ahead needs start_bar_max <= n_bars-2 "
                             f"({self.n_bars - 2}), got {self.start_bar_max}")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _hand_points(self, data) -> jnp.ndarray:
        """World coordinates of the two calibrated grasp points (L, R)."""
        pts = []
        for s in ("left", "right"):
            bid = self.wrist_body[s]
            pts.append(data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ self.seat_local[s])
        return jnp.concatenate(pts)

    def _grasp(self, data) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """(dist[2, n_bars], softmax over bars[2, n_bars])."""
        p = self._hand_points(data).reshape(2, 3)
        d = jnp.sqrt((p[:, None, 0] - self.bar_x[None, :]) ** 2
                     + (p[:, None, 2] - self.bar_z) ** 2)
        return d, jax.nn.softmax(-d / TAU_GRASP, axis=-1)

    def _contact_load(self, data) -> jnp.ndarray:
        """(2,) REAL per-hand contact load: Sum|qfrc_constraint| over finger DOFs.

        This is the only true contact signal MJX exposes (no ``data.contact`` in
        the Warp backend, no ``<touch>`` sensor in MJX -- see the CONTACT_F0 note).
        It says "this hand is carrying load through its fingers"; it does NOT say
        which bar, which is why every consumer ANDs it with a distance gate.
        """
        return jnp.stack([jnp.abs(data.qfrc_constraint[self.finger_dof[s]]).sum()
                          for s in ("left", "right")])

    # ---- M3.1 coverage / extrema metrics ---------------------------------- #
    # The evaluator (brax EvalWrapper) SUMS per-step metrics over an episode, so
    # every number here is built so that its episode SUM is interpretable:
    #   * "first crossing" indicators -> sum = 1 iff it ever happened (mean over
    #     envs = the probability we want, e.g. C(r) = P(min hand dist to B1 < r));
    #   * "improvement/monotone increment" -> sum = (constant - final min) or the
    #     running max, i.e. a true episode extremum instead of a mean.
    COVER_RADII = (0.50, 0.30, 0.20, 0.10)
    # running trackers live in info[] and start at these values
    _D0 = 2.0      # initial "min distance to B1" bound (nothing is 2 m away)

    def _coverage(self, data, info, hold_dual=0.0, h_park=0.0,
                  contact_now=None, near_goal=None, load=None):
        """Return (metrics_update, info_update) for the coverage diagnostics.

        `hold_dual` is passed in because it is computed in ``step`` (it is history,
        like the other streak features); everything else is derived from `data`.
        """
        d, w = self._grasp(data)
        c = jnp.exp(-(d.min(axis=-1) / TAU_CONTACT) ** 2)
        d_b1 = jnp.min(d[:, 1])                      # closest hand to bar 1
        prev_min = info["cov_min_d_b1"]
        cov_min_d_b1 = jnp.minimum(prev_min, d_b1)

        pos = self._hand_points(data).reshape(2, 3)
        x0 = self.bar_x[0]
        fwd = jnp.max(pos[:, 0]) - x0                # furthest hand beyond bar 0
        sway = jnp.abs(data.qpos[0] - x0)            # torso horizontal excursion
        speed = jnp.linalg.norm(data.qvel[:3])
        hold = (jnp.max(c) > 0.5).astype(jnp.float32)   # at least one hand on a bar
        # Per-hand grip, same criterion as `bar_L`/`bar_R` in step() (< GRASP_THRESH
        # on the nearest bar), so "both on" == bar_L >= 0 AND bar_R >= 0.
        dmin = jnp.min(d, axis=-1)
        on = dmin < GRASP_THRESH
        bar = jnp.where(on, jnp.argmax(w, axis=-1).astype(jnp.float32), -1.0)
        both = (on[0] & on[1]).astype(jnp.float32)
        # Bar-specific streak predicates.  bar 0 = the bar the episode starts on,
        # bar 1 = the next one (M3's B0 -> B1 task), whatever `goal_bar` is pinned to.
        # (Run #10 showed why the any-bar version is not enough: it saturates on the
        # initial two-handed hang on B0 and says nothing about the transfer.)
        hold_b1 = ((bar[0] == 1.0) & (bar[1] == 1.0)).astype(jnp.float32)
        handed = (((bar[0] == 0.0) & (bar[1] == 1.0))
                  | ((bar[0] == 1.0) & (bar[1] == 0.0))).astype(jnp.float32)
        air = ((~on[0]) & (~on[1])).astype(jnp.float32)

        def streak(cond, key):
            """Running consecutive-step counter + its all-time max (sum = max)."""
            run = jnp.where(cond > 0, info[key] + 1.0, 0.0)
            return run, jnp.maximum(0.0, run - info[key + "max"]), \
                jnp.maximum(info[key + "max"], run)

        # both hands inside the SAME bar's window (the "dual support" layer)
        dual_now = jnp.any((d[0] < GRASP_THRESH) & (d[1] < GRASP_THRESH)).astype(jnp.float32)
        run, both_imp, both_max = streak(both, "cov_both_run")
        run_dual, dual_imp, dual_max = streak(dual_now, "cov_dual_run")
        park_now = (jnp.any(jnp.abs(self.hang_x - data.qpos[0]) < PARK_RX)
                    & (jnp.abs(data.qpos[2] - self.key_qpos[2]) < PARK_RZ))
        _run_park, park_imp, _max_park = streak(park_now.astype(jnp.float32), "cov_park_run")
        run_b1, b1_imp, b1_max = streak(hold_b1, "cov_b1_run")
        run_xf, xf_imp, xf_max = streak(handed, "cov_xfer_run")
        run_air, air_imp, air_max = streak(air, "cov_air_run")
        # ---- 9.72 real-contact controls -------------------------------------
        # `near_goal` is the distance-only criterion the hnext family uses;
        # `contact_now` is the force-gated one.  Their difference IS the
        # "hovering at the instructed bar with the fingers open" measure, which is
        # the pathology this whole feature exists to remove -- so track all three
        # explicitly (positive control: keyframe hang; negative: released robot).
        if contact_now is None:
            contact_now = jnp.zeros((2,))
        if near_goal is None:
            near_goal = jnp.zeros((2,))
        if load is None:
            load = jnp.zeros((2,))
        hover = (near_goal * (1.0 - contact_now))
        run_ct, ct_imp, ct_max = streak(contact_now[0] * contact_now[1], "cov_contact_run")

        m = {
            # true episode minimum (sum = D0 - min) and the first-crossing P(min<r)
            "cov_min_d_b1_improve": jnp.maximum(0.0, prev_min - d_b1),
            "cov_swing_improve": jnp.maximum(0.0, sway * hold - info["cov_swing"]),
            "cov_fwd_improve": jnp.maximum(0.0, fwd - info["cov_fwd"]),
            "cov_speed_improve": jnp.maximum(0.0, speed - info["cov_speed"]),
            "cov_hand_on_steps": hold,
            # double support: sum = #steps with both hands gripping, and (via the
            # running-counter increment) sum = longest CONSECUTIVE such streak.
            # Includes the initial B0 hang, so it saturates early -- see cov_b1_*.
            "cov_both_on_steps": both,
            "cov_both_runmax_improve": both_imp,
            # dual support = both hands inside the SAME bar's window (the metric the
            # M4 discussion doc asks for; `cov_both_*` above is the weaker "each hand
            # is on some bar" version)
            "cov_dual_on_steps": dual_now,
            "cov_dual_runmax_improve": dual_imp,
            "cov_dual_hold_sum": jnp.asarray(hold_dual),
            "cov_park_on_steps": park_now.astype(jnp.float32),
            "cov_park_runmax_improve": park_imp,
            "cov_park_hold_sum": jnp.asarray(h_park),
            "cov_single_on_steps": (jnp.any(on) & (dual_now < 0.5)).astype(jnp.float32),
            # "grabbed the next bar with BOTH hands and stayed": sum = longest streak
            # of both hands on bar 1.  25 steps = 0.5 s = dwell_steps.
            "cov_b1_runmax_improve": b1_imp,
            # the actual transfer state: one hand still on bar 0 while the other is
            # on bar 1 (either handedness).  0 = the policy never supports the
            # hand-over and always lets go with both hands.
            "cov_xfer_runmax_improve": xf_imp,
            # how long it flies with NO hand on any bar (the "dive"); run #10 = ~11
            "cov_air_runmax_improve": air_imp,
            # 9.72: real-contact readouts, per hand and combined.  Acceptance
            # controls live in src/check_contact_sense.py.
            "cov_contact_L_on_steps": contact_now[0],
            "cov_contact_R_on_steps": contact_now[1],
            "cov_contact_both_on_steps": contact_now[0] * contact_now[1],
            "cov_contact_runmax_improve": ct_imp,
            # the distance-only criterion (== what hnext thresholds) and the
            # "near but NOT loaded" gap between them
            "cov_near_L_on_steps": near_goal[0],
            "cov_near_R_on_steps": near_goal[1],
            "cov_hover_L_on_steps": hover[0],
            "cov_hover_R_on_steps": hover[1],
            # raw EMA load, so the threshold can be re-read off any run
            "cov_load_L": load[0],
            "cov_load_R": load[1],
        }
        for r in self.COVER_RADII:
            tag = f"{int(r * 100):03d}"
            m[f"cov_cross_b1_{tag}"] = ((d_b1 < r) & (prev_min >= r)).astype(jnp.float32)

        info_upd = dict(cov_min_d_b1=cov_min_d_b1,
                        cov_swing=jnp.maximum(info["cov_swing"], sway * hold),
                        cov_fwd=jnp.maximum(info["cov_fwd"], fwd),
                        cov_speed=jnp.maximum(info["cov_speed"], speed),
                        cov_both_run=run, cov_both_runmax=both_max,
                    cov_dual_run=run_dual, cov_dual_runmax=dual_max,
                        cov_b1_run=run_b1, cov_b1_runmax=b1_max,
                        cov_xfer_run=run_xf, cov_xfer_runmax=xf_max,
                        cov_air_run=run_air, cov_air_runmax=air_max,
                        cov_contact_run=run_ct, cov_contact_runmax=ct_max,
                        cov_park_run=_run_park, cov_park_runmax=_max_park)
        # ⚠️ Return ONLY the coverage keys.  This function used to return a full
        # `dict(info)`, and `step` applied it as the LAST update -- which silently
        # reverted `max_bar`, `dwell`, `prev_bar` and `switches_total` to their
        # previous-step values (so the running `dwell` never got past 1 and
        # `dwell_success` could never fire; `max_bar` was the *instantaneous* bar
        # index, not the episode max).
        return m, info_upd

    def _cov_zero_metrics(self):
        keys = ["cov_min_d_b1_improve", "cov_swing_improve", "cov_fwd_improve",
                "cov_speed_improve", "cov_hand_on_steps",
                "cov_both_on_steps", "cov_both_runmax_improve",
                "cov_dual_on_steps", "cov_dual_runmax_improve",
                "cov_dual_hold_sum", "cov_single_on_steps",
                "cov_park_on_steps", "cov_park_runmax_improve", "cov_park_hold_sum",
                "cov_b1_runmax_improve", "cov_xfer_runmax_improve",
                "cov_air_runmax_improve",
                # 9.72 real-contact metrics (see _coverage)
                "cov_contact_L_on_steps", "cov_contact_R_on_steps",
                "cov_contact_both_on_steps", "cov_contact_runmax_improve",
                "cov_near_L_on_steps", "cov_near_R_on_steps",
                "cov_hover_L_on_steps", "cov_hover_R_on_steps",
                "cov_load_L", "cov_load_R"]
        keys += [f"cov_cross_b1_{int(r * 100):03d}" for r in self.COVER_RADII]
        return {k: jnp.zeros(()) for k in keys}

    def _cov_zero_info(self):
        return {"cov_min_d_b1": jnp.full((), self._D0), "cov_swing": jnp.zeros(()),
                "cov_fwd": jnp.full((), -self._D0), "cov_speed": jnp.zeros(()),
                "cov_both_run": jnp.zeros(()), "cov_both_runmax": jnp.zeros(()),
                "cov_b1_run": jnp.zeros(()), "cov_b1_runmax": jnp.zeros(()),
                "cov_xfer_run": jnp.zeros(()), "cov_xfer_runmax": jnp.zeros(()),
                "cov_air_run": jnp.zeros(()), "cov_air_runmax": jnp.zeros(()),
                "cov_dual_run": jnp.zeros(()), "cov_dual_runmax": jnp.zeros(()),
                "cov_contact_run": jnp.zeros(()), "cov_contact_runmax": jnp.zeros(()),
                "cov_park_run": jnp.zeros(()), "cov_park_runmax": jnp.zeros(()),
                # not a coverage metric, but this is where `reset` zero-initialises
                # the state-carried counters (used by the support_hold goal entry)
                "hold_streak": jnp.zeros(()),
                # the `hold` VALUE step actually used (it is the NEXT bar's streak
                # for "advance" and the any-bar one everywhere else -- recording the
                # resolved value here is what makes `_achieved_goal_info` exact for
                # every variant instead of re-deriving the choice and drifting)
                "hold_feature": jnp.zeros(()),
                # 9.72 contact-streak state (2,) + the EMA of the raw finger load
                "contact_streak": jnp.zeros((2,)),
                "contact_ema": jnp.zeros((2,)),
                # 9.74 bar-agnostic load trio streaks [any, dual, both]
                "lcontact_streak": jnp.zeros((3,))}

    def _state_features(self, data, prev_action: jnp.ndarray,
                        hold: jnp.ndarray = 0.0, kref: jnp.ndarray = 0.0,
                        progress: jnp.ndarray = 0.0,
                        hold_dual: jnp.ndarray = 0.0,
                        park: jnp.ndarray = 0.0,
                        k_goal: jnp.ndarray = 0.0,
                        hnext: jnp.ndarray = 0.0,
                        contact: jnp.ndarray = 0.0,
                        lcontact: jnp.ndarray = 0.0) -> jnp.ndarray:
        d, w = self._grasp(data)
        dmin = d.min(axis=-1)
        c = jnp.exp(-(dmin / TAU_CONTACT) ** 2)          # soft "grasping" indicator
        parts = [data.qpos[:7], data.qvel[:6], data.qpos[7:], data.qvel[6:],
                 self._hand_points(data), d.ravel(), w.ravel(), c]
        if self.goal_variant in SUPPORT_FAMILY:
            parts.append(jnp.max(c)[None])               # "at least one hand on"
            if self.goal_variant in HOLD_FAMILY:
                # `hold` is history (info["hold_streak"]) -- see HOLD_TAU.  It is in
                # the STATE as well as the goal on purpose: it changes "how far am I
                # from this goal", so hiding it would make the task a POMDP and alias
                # the critic (M4 discussion doc, section on state aliasing).
                parts.append(jnp.asarray(hold)[None])
            if self.goal_variant in DUAL_FAMILY:
                parts.append(jnp.asarray(hold_dual)[None])
        if self.goal_variant in CNEXT_FAMILY:
            parts.append(self._cnext(d, k_goal))
        if self.goal_variant in HPAIR_FAMILY:
            parts.append(jnp.broadcast_to(jnp.asarray(hnext, jnp.float32), (2,)))
        if self.goal_variant in DNEXT_FAMILY:
            parts.append(self._dnext(d, k_goal))
        if self.goal_variant in CONTACT_FAMILY:
            # `contact` is history (the streak of info["contact_streak"] mapped
            # through _hold_feature in `step`), exactly like `hnext`: it is a
            # function of the REAL finger load, not of the instantaneous state.
            parts.append(jnp.broadcast_to(jnp.asarray(contact, jnp.float32), (2,)))
        if self.goal_variant in LCONTACT_FAMILY:
            # 9.74: bar-agnostic load trio, also history (info["lcontact_streak"])
            parts.append(jnp.broadcast_to(jnp.asarray(lcontact, jnp.float32), (3,)))
        if self.goal_variant == "park":
            # only the parked streak is added: x_torso and z_torso are already
            # features 0 and 2 of the base block
            parts.append(jnp.asarray(park)[None])
        if self.goal_variant in CROSS_FAMILY:
            pp = self._hand_points(data).reshape(2, 3)
            rel = pp[1] - jnp.array([self.bar_x[CROSS_TARGET], 0.0, self.bar_z])
            cont = lambda d_: jnp.exp(-(d_ / TAU_CONTACT) ** 2)
            parts.append(rel)
            parts.append(jnp.sqrt(rel[0] ** 2 + rel[2] ** 2)[None])
            parts.append(cont(d[1, CROSS_TARGET])[None])    # c_RB1
            parts.append(cont(d[0, CROSS_SUPPORT])[None])   # c_LB0
            parts.append(cont(d[0, CROSS_TARGET])[None])    # c_LB1
            parts.append(cont(d[1, CROSS_SUPPORT])[None])   # c_RB0
        parts.append(prev_action)
        if self.goal_variant == "advance":
            parts.append(self._advance_features(data, kref, hold, progress))
            parts.append((jnp.asarray(kref) / max(self.n_bars - 1, 1))[None])
        return jnp.concatenate(parts)

    def _features_numpy(self, kd, action: np.ndarray) -> np.ndarray:
        """Goal features from a plain-MuJoCo data (used once, at init)."""
        p = []
        d = []
        for s in ("left", "right"):
            bid = self.wrist_body[s]
            pt = kd.xpos[bid] + kd.xmat[bid].reshape(3, 3) @ np.array(self.seat_local[s])
            p.append(pt)
            d.append(np.hypot(pt[0] - np.array(self.bar_x), pt[2] - self.bar_z))
        dmin = float(np.min(d))
        c = float(np.exp(-(dmin / TAU_CONTACT) ** 2))
        # superset [x, z, p_L(3), p_R(3), c_L, c_R, c_support, rel_pR(3), c_R_B1, c_L_B0]
        pR, pL = p[1], p[0]                               # right / left hand points
        rel = pR - np.array([self.bar_x[CROSS_TARGET], 0.0, self.bar_z])
        dd = np.hypot(pR[0] - self.bar_x[CROSS_TARGET], pR[2] - self.bar_z)
        dl0 = np.hypot(pL[0] - self.bar_x[CROSS_SUPPORT], pL[2] - self.bar_z)
        dl1 = np.hypot(pL[0] - self.bar_x[CROSS_TARGET], pL[2] - self.bar_z)
        dr0 = np.hypot(pR[0] - self.bar_x[CROSS_SUPPORT], pR[2] - self.bar_z)
        cont = lambda dd: float(np.exp(-(dd / TAU_CONTACT) ** 2))
        # The last entry (index 19) is the sustained-contact feature.  It is history,
        # so it cannot be derived from a single `data`; for the canonical goal we use
        # the *settled* value 1.0 ("a hang that has been going on for a while").
        # M4 tail = the GOAL vector, i.e. what `_advance_features` reads at the
        # *target* state ("hanging under the next bar, and staying there").  There
        # dx = qpos[0] - bar_x[next] equals the keyframe's offset from ITS bar (the
        # hang pose is the same at every bar), both hand-to-next-bar distances are 0,
        # dz is 0 and the sustained-contact entry is 1 (a settled hang).
        dx_goal = float(kd.qpos[0]) - float(self.bar_x[0])
        return np.concatenate([[kd.qpos[0], kd.qpos[2]], np.concatenate(p),
                               [c, c, c], rel, [dd, cont(dd), cont(dl0), cont(dl1),
                                                cont(dr0)], [1.0, 1.0],
                               # instruction goal: "arrive at the next bar and stay"
                               # (progress = 0, i.e. do not ask for the *next* advance;
                               # the relabelled training goals carry the real progress)
                               [dx_goal, 0.0, 0.0, 0.0, 1.0, 0.0],
                               # park goal: torso at the target bar's hang position
                               # (x/z come from goal0 + k*shift) and settled
                               [1.0],
                               # dual_cnext: both hands' grasp SEATS on the goal bar.
                               # 1.0 is the *keyframe* alignment (seat_local is built
                               # so the seat coincides with the bar centre at t=0); a
                               # settled hang sits 3.2 cm off -> 0.52, which is exactly
                               # what makes this the highest-gain dim in the goal.
                               [1.0, 1.0],
                               # dual_hnext: both hands settled (sustained) on the
                               # goal bar -- the same 1.0 convention as h_any/h_dual
                               [1.0, 1.0],
                               # dual6d: the grasp seats sit ON the goal bar at the
                               # canonical hang, so the per-hand distance is 0
                               [0.0, 0.0],
                               # 9.72 contact family: at the canonical hang both
                               # hands are clamped on the bar and carry 40 N m, so
                               # the sustained-contact-strength entry is settled at 1
                               [1.0, 1.0],
                               # 9.74 bar-agnostic load trio: same convention
                               [1.0, 1.0, 1.0]]).astype(np.float32)

    def _synergy(self, side: str, c: jnp.ndarray) -> jnp.ndarray:
        """Grasp synergy: closure c in [0,1] -> 7 finger joint targets."""
        base = self.key_ctrl[self.hand_act[side]]
        return c * base

    def _ctrl(self, action: jnp.ndarray) -> jnp.ndarray:
        ctrl = self.key_ctrl.at[self.act_armwaist].add(self.act_scale * action[:12])
        ctrl = jnp.clip(ctrl, jnp.array(self.mj_model.actuator_ctrlrange[:, 0]),
                        jnp.array(self.mj_model.actuator_ctrlrange[:, 1]))
        for i, s in enumerate(("left", "right")):
            c = 0.5 * (action[12 + i] + 1.0)
            ctrl = ctrl.at[self.hand_act[s]].set(self._synergy(s, c))
        return ctrl

    def _achieved_goal(self, data, hold: jnp.ndarray = 0.0,
                       kref: jnp.ndarray = 0.0, progress: jnp.ndarray = 0.0,
                       hold_dual: jnp.ndarray = 0.0,
                       park: jnp.ndarray = 0.0,
                       k_goal: jnp.ndarray = 0.0,
                       hnext: jnp.ndarray = 0.0,
                       contact: jnp.ndarray = 0.0,
                       lcontact: jnp.ndarray = 0.0) -> jnp.ndarray:
        # `hold` means "the sustained-contact scalar this variant uses":
        #   support_hold -> at least one hand on ANY bar
        #   advance      -> at least one hand on the NEXT bar
        # (see the streak bookkeeping in step()).
        d, _ = self._grasp(data)
        c = jnp.exp(-(d.min(axis=-1) / TAU_CONTACT) ** 2)
        p = self._hand_points(data)
        pR = p[3:6]
        rel = pR - jnp.array([self.bar_x[CROSS_TARGET], 0.0, self.bar_z])
        dd = jnp.sqrt(rel[0] ** 2 + rel[2] ** 2)          # |p_R - p_B1| in the x-z plane
        cont = lambda d_: jnp.exp(-(d_ / TAU_CONTACT) ** 2)
        full = jnp.concatenate([
            data.qpos[:1], data.qpos[2:3], p, c, jnp.max(c)[None],
            rel, dd[None],
            cont(d[1, CROSS_TARGET])[None],    # c_RB1
            cont(d[0, CROSS_SUPPORT])[None],   # c_LB0
            cont(d[0, CROSS_TARGET])[None],    # c_LB1
            cont(d[1, CROSS_SUPPORT])[None],   # c_RB0
            jnp.asarray(hold)[None],           # index 19: h_any (sustained contact)
            jnp.asarray(hold_dual)[None],                      # index 20 (h_dual)
            self._advance_features(data, kref, hold, progress),   # indices 21..26
            jnp.asarray(park)[None],                           # index 27 (h_park)
            self._cnext(d, k_goal),                            # indices 28, 29
            # (2,) for every variant: the default is a scalar 0.0, and the
            # superset must keep a fixed shape for the non-"dual_hnext" cases
            jnp.broadcast_to(jnp.asarray(hnext, jnp.float32), (2,)),   # 30, 31
            self._dnext(d, k_goal),                            # indices 32, 33
            # 9.72: the two real-contact sustained dims (see CONTACT_FAMILY)
            jnp.broadcast_to(jnp.asarray(contact, jnp.float32), (2,)),   # 34, 35
            # 9.74: bar-agnostic load trio
            jnp.broadcast_to(jnp.asarray(lcontact, jnp.float32), (3,))])  # 36, 37, 38
        return full[self._goal_full_idx]

    def _achieved_goal_info(self, data, info) -> jnp.ndarray:
        """`_achieved_goal` with every history-dependent argument taken from `info`.

        Single entry point for callers that hold a `State` instead of the raw
        streak arguments (``render_policy.py``).  It exists because the old
        render-side dispatch hand-assembled the arguments per variant, and got
        them wrong twice -- most recently ``uses_hpair`` was silently clobbered to
        ``variant == "dual_hnext"``, so every hnext-family variant except
        dual_hnext reported a goal distance computed with h_L,gk = h_R,gk = 0.
        """
        return self._achieved_goal(
            data,
            info["hold_feature"],
            info["k_ref"], info["k_ref"] - info["k_start"],
            self._hold_feature(info["dual_streak"]),
            self._hold_feature(info["park_streak"]),
            info["k_goal"],
            self._hold_feature(info["hpair_streak"]),
            self._hold_feature(info["contact_streak"]),
            self._hold_feature(info["lcontact_streak"]))

    def _cnext(self, d: jnp.ndarray, k_goal: jnp.ndarray) -> jnp.ndarray:
        """[c_L,gk, c_R,gk]: per-hand soft contact with the INSTRUCTION GOAL BAR.

        `d` is the (2, n_bars) hand-to-bar distance matrix from `_grasp`.  Single
        source of truth for both `_state_features` and `_achieved_goal` so the state
        and the goal can never disagree (same reason `_goal_full_idx` exists).
        """
        kg = jnp.clip(jnp.rint(jnp.asarray(k_goal)), 0.0,
                      self.n_bars - 1).astype(jnp.int32)
        return jnp.stack([jnp.exp(-(d[0, kg] / TAU_CONTACT) ** 2),
                          jnp.exp(-(d[1, kg] / TAU_CONTACT) ** 2)])

    def _dnext(self, d: jnp.ndarray, k_goal: jnp.ndarray) -> jnp.ndarray:
        """[d_L,gk, d_R,gk]: per-hand DISTANCE to the instruction goal bar (metres).

        The linear counterpart of `_cnext`: unlike exp(-(d/tau)^2) it keeps a
        gradient at any range, so it can drive a hand that starts far away.
        """
        kg = jnp.clip(jnp.rint(jnp.asarray(k_goal)), 0.0,
                      self.n_bars - 1).astype(jnp.int32)
        return jnp.stack([d[0, kg], d[1, kg]])

    def _hold_feature(self, streak: jnp.ndarray) -> jnp.ndarray:
        """Consecutive gripping steps -> the sustained-contact goal entry."""
        return 1.0 - jnp.exp(-jnp.asarray(streak) / HOLD_TAU)

    def _advance_features(self, data, kref, hold_next=0.0, progress=0.0) -> jnp.ndarray:
        """[dx, dz, d_L,next, d_R,next, hold_next, progress] vs bar k_ref+1.

        Translation invariant (dx/dz are differences) and progress invariant (the
        reference bar travels with the robot), so one goal covers every bar.

        `hold_next` is the M4 analogue of the M3 `hold` term that run #11 showed to
        be essential: the two contact entries above are *instantaneous*, so without
        it a ballistic pass through "under the next bar, both hands touching it"
        would satisfy the goal for a few frames (exactly the exploit that made run
        #10 graze instead of grasp).  `hold_next` only rises while a hand stays in
        the next bar's window, so the target state is "arrived AND stayed".
        """
        d, _ = self._grasp(data)
        k = jnp.clip(jnp.asarray(kref) + 1.0, 0.0, self.n_bars - 1).astype(jnp.int32)
        x_bar = jnp.take(self.bar_x, k)
        dx = data.qpos[0] - x_bar
        dz = data.qpos[2] - self.key_qpos[2]
        # Per-hand DISTANCE to the next bar (metres), not the saturating soft
        # contact: as run #12 showed, a translation-invariant goal whose only
        # task-relevant entries are constants-until-contact gives the actor no
        # direction to move in (the goal is (nearly) equal to the present state for
        # a stably hanging robot -> the contrastive task degenerates, acc 0.04-0.10,
        # and the policy just hangs on its start bar).  A distance decreases
        # monotonically as a hand reaches out, so it restores both the gradient and
        # the "reach with one hand first" precursor, while staying translation
        # invariant (bar-independent).
        d_next = d[:, k]
        return jnp.concatenate([dx[None], dz[None], d_next,
                                jnp.asarray(hold_next)[None],
                                jnp.asarray(progress)[None]])

    def episode_info_zero(self):
        """Reset values of the episode-scoped `info` entries at t=0.

        Two users: ``reset()`` and ``MjxAutoResetWrapper`` (brax's auto-reset only
        swaps ``pipeline_state``/``obs``, so without this the counters below kept
        their values across episode boundaries, e.g. `cov_b1_runmax` could exceed
        the number of double-support steps in the same episode).
        """
        return {"max_bar": jnp.zeros(()), "dwell": jnp.zeros(()),
                "prev_bar": jnp.full((2,), -1.0), "switches_total": jnp.zeros(()),
                "k_ref": jnp.zeros(()), "k_ref_run": jnp.zeros(()),
                "k_ref_run_bar": jnp.full((), -1.0),
                "kref_max": jnp.zeros(()),
                "k_start": jnp.zeros(()),
                "next_streak": jnp.zeros(()),
                "dual_streak": jnp.zeros(()),
                "park_streak": jnp.zeros(()),
                "k_goal": jnp.zeros(()),        # instruction goal bar (dual_cnext)
                "hpair_streak": jnp.zeros((2,)),  # per-hand sustained contact with it
                **self._cov_zero_info()}

    # ------------------------------------------------------------------ #
    # brax Env interface
    # ------------------------------------------------------------------ #

    def reset(self, rng: jax.Array) -> State:
        rng_pos, rng_arm, rng_leg, rng_vel, rng_goal = jax.random.split(rng, 5)
        nr, na, nl, nv = self._reset_cfg

        # perturb the free joint (small) and the joints (arms small, legs a bit larger)
        root = jax.random.uniform(rng_pos, (3,), minval=-nr, maxval=nr)
        qpos = self.key_qpos
        jn = jnp.arange(self.njoints)
        noise = jnp.where(jn < 12,
                          jax.random.uniform(rng_leg, (self.njoints,), minval=-nl, maxval=nl),
                          jax.random.uniform(rng_arm, (self.njoints,), minval=-na, maxval=na))
        qpos = qpos.at[:3].add(root).at[7:].add(noise)
        # NB: `k0` and the sampled goal bar must come from DIFFERENT keys.  Reusing
        # `rng_goal` for both makes them strongly correlated (the same uniform bits
        # decide both draws), which in practice pinned gk to k0+1 -- i.e. the "goal is
        # any later bar" distribution silently collapsed to "exactly the next bar".
        rng_bar, rng_goal_sel = jax.random.split(rng_goal)
        k0 = jnp.zeros(())          # bar the episode starts on (0 unless randomised)
        gk_sample = None            # set here when the curriculum table is used
        if self.start_bar_max > 0:
            # The bars are periodic, so "start on bar k" is just a translation of the
            # whole robot by k * spacing -- bit-identical physics, no new asset.
            if self.goal_bar is None:
                # delta-first curriculum sampling (see the table built in __init__)
                i = jax.random.randint(rng_goal_sel, (), 0, self._delta.shape[0])
                d = jnp.take(self._delta, i)
                lo, hi = jnp.take(self._klo, i), jnp.take(self._khi, i)
                k0 = jax.random.randint(rng_bar, (), lo, hi + 1).astype(jnp.float32)
                gk_sample = k0.astype(jnp.int32) + d
            else:
                k0 = jax.random.randint(rng_bar, (), 0,
                                        self.start_bar_max + 1).astype(jnp.float32)
            qpos = qpos.at[0].add(k0 * self.bar_spacing)
        qvel = nv * jax.random.normal(rng_vel, (self.mj_model.nv,))

        data = self._data0.replace(qpos=qpos, qvel=qvel, ctrl=self.key_ctrl)
        # settle a few physics steps so the perturbed grasp re-seats before the episode starts
        for _ in range(self.n_frames):
            data = mjx.step(self.mx_model, data)

        if self.goal_variant in CROSS_FAMILY:
            gk = jnp.asarray(0)          # single fixed goal (see goal_set above)
        elif self.goal_bar is None:
            if self.goal_ahead:
                # goal bar strictly AFTER the bar this episode started on (M4:
                # "start anywhere in the first N-1 bars, the goal is any later bar")
                k0i = k0.astype(jnp.int32)
                span = jnp.maximum(
                    jnp.minimum(self.n_bars - 1 - k0i, self.goal_bar_max - k0i), 1)
                if self.goal_ahead_max > 0:
                    # cap the distance: --goal-ahead-max 1 means "the goal is ALWAYS
                    # exactly the next bar", i.e. a uniformly difficult, always
                    # reachable instruction goal (run #14 showed that goals 2-4 bars
                    # ahead destroy the transfer instead of extending it).
                    span = jnp.minimum(span, self.goal_ahead_max)
                gk = jnp.minimum(k0i + 1 + jax.random.randint(rng_goal_sel, (), 0, span),
                                 self.n_bars - 1)
            elif gk_sample is not None:
                # already drawn together with k0 (delta-first curriculum sampling)
                gk = gk_sample
            else:
                gk = jax.random.randint(rng_goal, (), self.goal_bar_min, self.goal_bar_max + 1)
        else:
            gk = jnp.asarray(self.goal_bar)
        goal = self.goal_set[gk]

        # `hold` starts at 0: the streak counter has no history at t=0, so even
        # though both hands are gripping B0 the sustained-contact entry is 0 here
        # and only rises over the next few steps (see HOLD_TAU).
        obs = jnp.concatenate(
            [self._state_features(data, jnp.zeros(14), self._hold_feature(0.0), 0.0,
                                  0.0, self._hold_feature(0.0), self._hold_feature(0.0),
                                  jnp.asarray(gk, dtype=jnp.float32),
                                  self._hold_feature(jnp.zeros((2,))),
                                  self._hold_feature(jnp.zeros((2,))),
                                  self._hold_feature(jnp.zeros((3,)))), goal])
        # JaxGCRL's evaluator hardcodes these five names:
        #   reward, success, success_easy, dist, distance_from_origin
        # (see jaxgcrl/utils/evaluator.py).  Everything else is our own.
        metrics = {k: jnp.zeros(()) for k in
                   ("reward", "success", "success_easy", "dist", "distance_from_origin",
                    "dwell_success", "max_bar", "bar_L", "bar_R", "hand_switches",
                    "fell", "forward", "advance_max")}
        metrics.update(self._cov_zero_metrics())
        info = {"goal": goal, **self.episode_info_zero()}
        # the instruction bar (float, so it can ride in `info`); read by the
        # "dual_cnext" contact features.  Constant within an episode.
        info["k_goal"] = jnp.asarray(gk, dtype=jnp.float32)
        if self.goal_variant == "advance":
            # The reference/progress origin is the bar the episode actually starts
            # on -- with --start-bar-max > 0 that is not 0, and leaving k_ref at 0
            # would make the first goal point AT (or behind) the robot.
            info["k_start"] = k0
            info["k_ref"] = k0
        return State(pipeline_state=data, obs=obs, reward=jnp.zeros(()), done=jnp.zeros(()),
                     metrics=metrics, info=info)

    def step(self, state: State, action: jax.Array) -> State:
        data = state.pipeline_state
        ctrl = self._ctrl(action)
        data = data.replace(ctrl=ctrl)
        for _ in range(self.n_frames):
            data = mjx.step(self.mx_model, data)

        d, w = self._grasp(data)
        dmin = d.min(axis=-1)
        c = jnp.exp(-(dmin / TAU_CONTACT) ** 2)
        bar = jnp.where(dmin < GRASP_THRESH, jnp.argmax(w, axis=-1).astype(jnp.float32), -1.0)

        # --- M4 progress index: the bar of the last *sustained* grip -----------
        # `bar` is per hand, >= 0 only inside the 5 cm grasp window; a grip must last
        # K_SUSTAIN steps before it advances the reference bar, so grazing the next
        # bar in passing does not count as progress.
        k_now = jnp.max(bar)
        same = (k_now == state.info["k_ref_run_bar"]) & (k_now >= 0)
        k_run = jnp.where(same, state.info["k_ref_run"] + 1.0,
                          jnp.where(k_now >= 0, 1.0, 0.0))
        k_run_bar = jnp.where(same, state.info["k_ref_run_bar"], k_now)
        k_ref = jnp.where((k_run >= K_SUSTAIN) & (k_run_bar > state.info["k_ref"]),
                          k_run_bar, state.info["k_ref"])
        kref_max = jnp.maximum(state.info["kref_max"], k_ref)

        # sustained contact: consecutive steps with at least one hand inside the
        # grasp window of some bar (dmin < GRASP_THRESH -- the same criterion as
        # `bar_L`/`bar_R`/`max_bar`, NOT the tighter `max(c) > 0.5` of _coverage:
        # a *settled* hang sits at dmin ~3.5-4.5 cm, so max(c) > 0.5 would call a
        # validated hang "not gripping").  The streak resets the moment no hand is
        # in any window, which is exactly what a ballistic dive does.
        gripping = (jnp.min(d) < GRASP_THRESH).astype(jnp.float32)
        hold_streak = jnp.where(gripping > 0, state.info["hold_streak"] + 1.0, 0.0)
        # M4: the same idea, but only for the bar the robot is advancing TO.  It has
        # to be k_ref + 1 (the bar the goal currently points at), NOT "the bar a hand
        # is on" -- otherwise the streak tracks the wrong bar the moment the robot
        # arrives, and hold_next never builds.
        k_next = jnp.clip(k_ref + 1.0, 0.0, self.n_bars - 1).astype(jnp.int32)
        on_next = (jnp.min(d[:, k_next]) < GRASP_THRESH).astype(jnp.float32)
        next_streak = jnp.where(on_next > 0, state.info["next_streak"] + 1.0, 0.0)
        hold = (self._hold_feature(next_streak) if self.goal_variant == "advance"
                else self._hold_feature(hold_streak))
        # h_dual: both hands inside the SAME bar's 5 cm window (5 cm = the env's own
        # grasp criterion, NOT the tighter max(c)>0.5 = 3.3 cm: a settled hang drifts
        # out to 3-4.5 cm and would flicker the streak to zero)
        in_win = d < GRASP_THRESH
        dual_now = jnp.any(in_win[0] & in_win[1])
        dual_streak = jnp.where(dual_now, state.info["dual_streak"] + 1.0, 0.0)
        # "dual_hnext": per-hand sustained contact with the GOAL bar.  Unlike
        # h_any (satisfied by the start bar) and h_dual (any single bar), these
        # only rise while THAT hand stays in the goal bar's window, so a robot
        # parked on B0 scores 0 and there is no "wait for h to grow" shortcut.
        kg_h = jnp.clip(jnp.rint(state.info["k_goal"]), 0.0,
                        self.n_bars - 1).astype(jnp.int32)
        in_goal = (d[:, kg_h] < GRASP_THRESH).astype(jnp.float32)
        hpair_streak = jnp.where(in_goal > 0, state.info["hpair_streak"] + 1.0, 0.0)
        hpair = self._hold_feature(hpair_streak)
        hold_dual = self._hold_feature(dual_streak)
        # h_park: consecutive steps with the TORSO inside some bar's hang box.  A
        # ballistic pass through the region lasts only a few steps, so a streak of
        # ~10 steps (0.2 s) already distinguishes "parked (i.e. supported)" from
        # "flew past" -- and unlike a hand-contact proxy it needs no hand
        # calibration (see 9.48).
        parked = (jnp.any(jnp.abs(self.hang_x - data.qpos[0]) < PARK_RX)
                  & (jnp.abs(data.qpos[2] - self.key_qpos[2]) < PARK_RZ))
        park_streak = jnp.where(parked, state.info["park_streak"] + 1.0, 0.0)
        h_park = self._hold_feature(park_streak)
        # ---- 9.72: the REAL contact signal ---------------------------------
        # `load` is the EMA of the raw per-hand finger load, and `contact_now` is
        # the AND of "this hand is inside the GOAL bar's window" (the geometric
        # gate -- the load alone cannot say which bar) and "this hand is actually
        # carrying load".  The distance-only counterpart `in_goal` above is what
        # the hnext family thresholds, so keeping both lets a run report exactly
        # how much of its "grip" is a real one (cov_hover_* = near AND !loaded).
        load_raw = self._contact_load(data)
        alpha = 1.0 / self.contact_ema
        load = state.info["contact_ema"] + alpha * (load_raw - state.info["contact_ema"])
        near_goal = (d[:, kg_h] < GRASP_THRESH).astype(jnp.float32)
        loaded = (load > self.contact_thresh).astype(jnp.float32)
        contact_now = near_goal * loaded
        contact_streak = jnp.where(contact_now > 0, state.info["contact_streak"] + 1.0, 0.0)
        contact = self._hold_feature(contact_streak)
        # ---- 9.74: the BAR-AGNOSTIC load trio --------------------------------
        # `grip_now[h]` = "hand h is really loaded, on whichever bar it is on":
        # the distance gate uses the hand's NEAREST bar (dmin), not the instruction
        # bar, which is what makes this family instruction-independent (so
        # --train-goal-bar no longer changes the training distribution) and what
        # keeps the *stepping stone* paying -- the goal=B2 dive of
        # `support_dual_contact` came precisely from aiming the load term at B2.
        grip_now = ((dmin < GRASP_THRESH) & (load > self.contact_thresh)).astype(jnp.float32)
        same_bar = jnp.any((d[0] < GRASP_THRESH) & (d[1] < GRASP_THRESH))
        l_now = jnp.stack([(jnp.max(grip_now) > 0).astype(jnp.float32),
                           (same_bar & (jnp.min(grip_now) > 0)).astype(jnp.float32),
                           (jnp.min(grip_now) > 0).astype(jnp.float32)])
        lcontact_streak = jnp.where(l_now > 0, state.info["lcontact_streak"] + 1.0, 0.0)
        lcontact = self._hold_feature(lcontact_streak)

        # progress = how many bars this episode has advanced (raw count, so one
        # bar is worth ~1.0 in the goal distance -- the same order as a contact
        # entry).  Without it a *stably hanging* state and a future state one
        # "progress level" later have identical relative features, so the majority
        # of relabelled goals would be trivially matched and nothing would push the
        # policy to keep going.
        progress = k_ref - state.info["k_start"]
        feats = self._state_features(data, action, hold, k_ref, progress, hold_dual,
                                     h_park, state.info["k_goal"], hpair, contact,
                                     lcontact)
        goal = state.info["goal"]
        obs = jnp.concatenate([feats, goal])

        achieved = self._achieved_goal(data, hold, k_ref, progress, hold_dual, h_park,
                                       state.info["k_goal"], hpair, contact, lcontact)
        dist = jnp.linalg.norm(achieved - goal)
        success = (dist < self.goal_reach_thresh).astype(jnp.float32)
        slow = jnp.linalg.norm(data.qvel) < 1.0
        dwell = jnp.where((success > 0) & slow, state.info["dwell"] + 1.0, 0.0)

        max_bar = jnp.maximum(state.info["max_bar"], jnp.max(bar))
        switched = jnp.any(bar != state.info["prev_bar"])
        fell = (data.qpos[2] < self.bar_z - FALL_DEPTH).astype(jnp.float32)

        metrics = dict(state.metrics)
        metrics.update(
            reward=jnp.zeros(()),                     # CRL is reward-free; key required by the evaluator
            success=success,
            success_easy=(dist < 3.0 * self.goal_reach_thresh).astype(jnp.float32),
            distance_from_origin=jnp.linalg.norm(data.qpos[:2]),
            forward=data.qpos[0] - self.bar_x[0],
            dwell_success=(dwell >= self.dwell_steps).astype(jnp.float32),
            dist=dist,
            max_bar=max_bar,
            bar_L=bar[0],
            bar_R=bar[1],
            # per-step signal (0/1) so any evaluator aggregation is unambiguous;
            # the running total lives in info["switches_total"]
            hand_switches=switched.astype(jnp.float32),
            fell=fell,
            # M4 headline: how many bars this episode ADVANCED (0 = never left the
            # start bar, 1 = one bar, ...), i.e. relative to where it started.
            advance_max=jnp.maximum(kref_max - state.info["k_start"], 0.0),
        )
        # NB: carry the existing info/metrics dicts forward — JaxGCRL's
        # TrajectoryIdWrapper and brax's EpisodeWrapper add keys (traj_id, steps,
        # truncation, episode metrics) and a changed pytree structure breaks the
        # lax.scan inside EpisodeWrapper.
        cov_m, cov_i = self._coverage(data, state.info, hold_dual, h_park,
                                      contact_now=contact_now, near_goal=near_goal,
                                      load=load)
        metrics.update(cov_m)

        info = dict(state.info)
        info.update(cov_i)                 # coverage trackers only (see _coverage)
        # ... and apply the authoritative values LAST, so no later update can
        # revert a running counter to its previous value.
        info.update(goal=goal, max_bar=max_bar, dwell=dwell, prev_bar=bar,
                    hold_streak=hold_streak, next_streak=next_streak,
                    dual_streak=dual_streak, park_streak=park_streak,
                    hpair_streak=hpair_streak,
                    contact_streak=contact_streak, contact_ema=load,
                    lcontact_streak=lcontact_streak,
                    # the resolved `hold` value this step used (variant-dependent
                    # source, so record it rather than re-derive it on read)
                    hold_feature=hold,
                    k_ref=k_ref, k_ref_run=k_run, k_ref_run_bar=k_run_bar,
                    kref_max=kref_max,
                    switches_total=(state.info.get("switches_total", jnp.zeros(()))
                                    + switched.astype(jnp.float32)))
        return State(pipeline_state=data, obs=obs, reward=jnp.zeros(()), done=fell,
                     metrics=metrics, info=info)

    @property
    def observation_size(self) -> int:
        return self.state_dim + len(self.goal_indices)

    @property
    def action_size(self) -> int:
        return 14

    @property
    def backend(self) -> str:
        return f"mjx-{self.impl}"


def create_brachiation(backend: Optional[str] = None, **kwargs) -> Brachiation:
    """Factory used by the smoke test / training entry point.

    ``backend`` mirrors JaxGCRL's ``create_env(env_name, backend=...)`` signature
    and is mapped onto the MJX implementation (``"jax"`` or ``"warp"``).
    """
    scene = kwargs.pop("scene", "full")
    if backend is not None:
        kwargs.setdefault("impl", "warp" if backend in ("warp", "mjx-warp") else "jax")
    paths = {
        "full": "assets/g1_brachiation/scene_bars.xml",
        # Easy-geometry variant: identical to "full" except bar spacing 0.35 m
        # (the builder re-solves the hang keyframe, so only the bar gaps change).
        # M3.0 measured static reach |p_R - B1| = 0.031 m at 0.40 vs 0.0007 m at 0.35.
        "full035": "assets/g1_brachiation/scene_bars_d035.xml",
        "mesh": "assets/g1_brachiation/scene_bars_mjx.xml",
        "allprim": "assets/g1_brachiation/scene_bars_mjx_allprim.xml",
    }
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return Brachiation(os.path.join(repo, paths[scene]), **kwargs)
