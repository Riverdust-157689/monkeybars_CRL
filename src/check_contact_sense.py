"""Acceptance controls for the 9.72 "real contact" feature (positive + negative).

The feature (``CONTACT_FAMILY``) is only worth training on if the underlying
signal really separates a loaded hand from a hand that merely *looks* loaded:

    f_{h,gk} = 1 - exp(-s_h / HOLD_TAU)
    s_h      = consecutive steps of  (d[h, gk] < GRASP_THRESH) AND (F_h > thresh)

with ``F_h`` = EMA(alpha = 1/CONTACT_EMA) of ``Sum|qfrc_constraint|`` over hand
``h``'s 7 finger DOFs and ``thresh = CONTACT_F0 * CONTACT_THETA``.

Four arms, all driven through the REAL ``env.step`` (action 0 on the 12 arm/waist
dims reproduces the keyframe arm ctrl exactly, and ``a[12+i] = 2c - 1``
reproduces the grasp synergy for closure ``c``), so nothing here bypasses the code
path training uses and no rendering is involved:

  P   both hands CLOSED, goal bar = B0
      positive control: ``contact`` must be 1 on BOTH hands for >= 95% of the
      steps, and the per-step EMA load must stay >= 2x the threshold.
  H   LEFT hand open, RIGHT hand closed, goal bar = B0
      the HOVER control, and the whole point of the feature: the robot hangs on
      its right hand, so the left hand stays parked inside B0's 5 cm window
      without gripping.  This is the physical counterpart of the observed goal=B1
      failure ("the left hand gets to B1 and just will not close") and of the
      dual_hnext late regression (right hand 0.000 m from B1 for 413 steps, grasp
      closure 0.12).  Expected: near_L == 1 for a stretch of steps while
      contact_L == 0 -- i.e. hnext's distance criterion scores that state as a
      grip and the force criterion does not.
  N   both hands OPEN, goal bar = B0
      negative control: the robot falls and, once the release has washed out
      (--skip 15), ``contact`` must be 0 and the finger load must stay below
      threshold/2 for EVERY recorded step.
  P2  both hands CLOSED, goal bar = B1 (the robot starts on B0)
      bar-identity control: the fingers carry a full body weight but on the
      WRONG bar, so ``contact`` must be 0 for every step while the load stays
      >= 2x the threshold.  (The raw force alone cannot say which bar a hand is
      loaded on -- that is why the feature ANDs it with the distance gate.)

``--skip`` steps are run but not recorded: the EMA is initialised to 0 at t=0, so
its first ~3 steps are a cold start (it reads 0.2 x raw).  Every steady-state
claim is therefore made on the post-skip window.  Measured cold start in the
positive control: 13.8 N m at t=0 vs 67 at t>=5, still above the 12 N m
threshold, so the feature never misreports during it.

Usage:  JAX_PLATFORMS=cpu .venv-warp/bin/python src/check_contact_sense.py
Exit code 0 = all controls passed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "third_party", "jaxgcrl"))

KEYS = ("cov_contact_L_on_steps", "cov_contact_R_on_steps",
        "cov_near_L_on_steps", "cov_near_R_on_steps",
        "cov_hover_L_on_steps", "cov_hover_R_on_steps",
        "cov_load_L", "cov_load_R")


def streak_feature_peak(x: np.ndarray, tau: float = 10.0) -> float:
    """Peak of 1-exp(-streak/tau) over a per-step 0/1 signal.

    The PEAK, not the final value: a hand that sits in the window for 11 steps
    and then leaves reaches 0.67 and returns to 0, and 0.67 is the number that
    says how close hnext's goal dim got to being satisfied.
    """
    s, best = 0.0, 0.0
    for v in x:
        s = s + 1.0 if v > 0 else 0.0
        best = max(best, 1.0 - np.exp(-s / tau))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="full035",
                    choices=["full", "full035", "mesh", "allprim"])
    ap.add_argument("--variant", default="support_dual_contact")
    ap.add_argument("--steps", type=int, default=30, help="recorded steps per arm")
    ap.add_argument("--closed", type=float, default=0.95,
                    help="min fraction of steps a closed hand must read loaded")
    ap.add_argument("--open", type=float, default=0.02,
                    help="max fraction of steps an unloaded hand may read loaded")
    ap.add_argument("--hover-peak", type=float, default=0.30,
                    help="min PEAK of hnext's 1-exp(-streak/10) over the hover "
                         "window -- i.e. how much the distance criterion is fooled")
    ap.add_argument("--trace", type=int, default=0,
                    help="print the first N recorded per-step rows of every arm")
    ap.add_argument("--contact-f0", type=float, default=None,
                    help="override CONTACT_F0 (default: the env's constant)")
    ap.add_argument("--contact-theta", type=float, default=None,
                    help="override CONTACT_THETA -- use it to show that the "
                         "controls hold over a band, not just at one value")
    ap.add_argument("--contact-ema", type=float, default=None,
                    help="override CONTACT_EMA (steps)")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from envs.brachiation import Brachiation

    scene = os.path.join(REPO, "assets", "g1_brachiation",
                         {"full": "scene_bars.xml", "full035": "scene_bars_d035.xml",
                          "mesh": "scene_bars_mjx.xml",
                          "allprim": "scene_bars_mjx_allprim.xml"}[args.scene])

    def run(goal_bar: int, closure_L: float, closure_R: float, skip: int):
        """Keyframe pose driven through env.step; returns the recorded arrays."""
        env_kw = {}
        if args.contact_f0 is not None:
            env_kw["contact_f0"] = args.contact_f0
        if args.contact_theta is not None:
            env_kw["contact_theta"] = args.contact_theta
        if args.contact_ema is not None:
            env_kw["contact_ema"] = args.contact_ema
        env = Brachiation(scene_xml=scene, impl="warp", goal_variant=args.variant,
                          goal_bar=goal_bar, n_bars=5, **env_kw)
        act = np.zeros(14, dtype=np.float32)
        act[12] = 2.0 * closure_L - 1.0      # ctrl the fingers via the grasp synergy
        act[13] = 2.0 * closure_R - 1.0
        step = jax.jit(env.step)
        state = env.reset(jax.random.PRNGKey(0))
        rec = {k: np.zeros(args.steps, dtype=np.float32) for k in KEYS}
        rec["z"] = np.zeros(args.steps, dtype=np.float32)
        skip = max(0, int(skip))
        for t in range(skip + args.steps):
            state = step(state, jnp.asarray(act))
            if t < skip:
                continue
            i = t - skip
            for k in KEYS:
                rec[k][i] = float(state.metrics[k])
            rec["z"][i] = float(state.pipeline_state.qpos[2])
        rec["final_streak"] = np.asarray(state.info["contact_streak"])
        rec["final_ema"] = np.asarray(state.info["contact_ema"])
        # the two dims the actor actually consumes: obs = [state | goal], and the
        # contact slots live in the state block for this family
        rec["obs_contact"] = np.asarray(state.obs[env.layout.contact])
        rec["env"] = env
        rec["skip"] = skip
        return rec

    # `skip` only matters where the EMA cold start would be recorded: H must see
    # t=0 (the hover begins immediately), P/P2 only need a settled filter, and N
    # must be past the release transient for its floor to mean anything.
    P = run(goal_bar=0, closure_L=1.0, closure_R=1.0, skip=5)
    H = run(goal_bar=0, closure_L=0.0, closure_R=1.0, skip=0)
    N = run(goal_bar=0, closure_L=0.0, closure_R=0.0, skip=15)
    P2 = run(goal_bar=1, closure_L=1.0, closure_R=1.0, skip=5)

    env0 = P["env"]
    thr = env0.contact_thresh

    def m(r, k):
        return float(np.mean(r[k]))

    print(f"[contact] variant={args.variant} scene={args.scene} "
          f"steps={args.steps} (after a per-arm skip) "
          f"f0={env0.contact_f0} theta={env0.contact_theta} -> thresh={thr:.1f} N.m  "
          f"ema={env0.contact_ema} steps")
    print("[contact] mean per-step fractions over the recorded window:")
    for tag, r in (("P  closed  both,  goal=B0", P),
                   ("H  L open  R closed, B0", H),
                   ("N  released,          B0", N),
                   ("P2 closed  both,  goal=B1", P2)):
        print(f"  {tag:26s} skip={r['skip']:2d}  "
              f"contact L/R = {m(r,'cov_contact_L_on_steps'):.2f} / "
              f"{m(r,'cov_contact_R_on_steps'):.2f}   "
              f"near L/R = {m(r,'cov_near_L_on_steps'):.2f} / "
              f"{m(r,'cov_near_R_on_steps'):.2f}   "
              f"hover L/R = {m(r,'cov_hover_L_on_steps'):.2f} / "
              f"{m(r,'cov_hover_R_on_steps'):.2f}")
    print("[contact] EMA finger load (N.m) over the recorded window:")
    for tag, r in (("P", P), ("H", H), ("N", N), ("P2", P2)):
        for h in ("L", "R"):
            v = r[f"cov_load_{h}"]
            print(f"  {tag:3s} {h}: min {v.min():6.1f}  p50 "
                  f"{np.percentile(v, 50):6.1f}  max {v.max():6.1f}")
    print(f"[contact] obs[contact] (what the actor sees) at the last step: "
          f"P={np.round(P['obs_contact'],3)}  H={np.round(H['obs_contact'],3)}  "
          f"N={np.round(N['obs_contact'],3)}  P2={np.round(P2['obs_contact'],3)}")
    hnext_peak = streak_feature_peak(H["cov_near_L_on_steps"])
    force_peak = streak_feature_peak(H["cov_contact_L_on_steps"])
    print("[contact] the HOVER gap, arm H (left hand parked at B0, not gripping; "
          f"{int(np.sum(H['cov_hover_L_on_steps']))} of {args.steps} steps are "
          "'near but not loaded'):")
    print(f"    hnext_L (DISTANCE criterion, peak 1-exp(-streak/10)) reaches "
          f"{hnext_peak:.3f}")
    print(f"    f_L,B0  (FORCE criterion,    peak 1-exp(-streak/10)) reaches "
          f"{force_peak:.3f}")

    if args.trace > 0:
        n = min(args.trace, args.steps)
        for tag, r in (("P", P), ("H", H), ("N", N), ("P2", P2)):
            print(f"[trace] arm {tag} (skip={r['skip']}): step near_L near_R "
                  f"load_L load_R cont_L cont_R hover_L    z")
            for t in range(n):
                print(f"          {t:3d}   {r['cov_near_L_on_steps'][t]:.0f}     "
                      f"{r['cov_near_R_on_steps'][t]:.0f}    "
                      f"{r['cov_load_L'][t]:6.1f} {r['cov_load_R'][t]:6.1f}  "
                      f"{r['cov_contact_L_on_steps'][t]:.0f}     "
                      f"{r['cov_contact_R_on_steps'][t]:.0f}     "
                      f"{r['cov_hover_L_on_steps'][t]:.0f}    {r['z'][t]:6.3f}")

    # ---- assertions ---------------------------------------------------------
    P_min = min(float(P[f"cov_load_{h}"].min()) for h in ("L", "R"))
    N_max = max(float(N[f"cov_load_{h}"].max()) for h in ("L", "R"))
    H_L_max = float(H["cov_load_L"].max())
    # arm H records from t=0, so its supporting hand still shows the EMA cold
    # start (0.2 x raw = 12.2 at t=0); the steady-state claim starts after it
    H_R_min = float(H["cov_load_R"][3:].min())
    checks = []
    for h in ("L", "R"):
        ck = f"cov_contact_{h}_on_steps"
        checks.append((f"P: closed hand {h} loaded >= {args.closed:.2f} of the steps",
                       m(P, ck) >= args.closed))
        checks.append((f"P2: hand {h} on the WRONG bar loaded <= {args.open:.2f}",
                       m(P2, ck) <= args.open))
        checks.append((f"P2: ... while its load stays >= 2x threshold "
                       f"(min {float(P2[f'cov_load_{h}'].min()):.1f} N.m)",
                       float(P2[f"cov_load_{h}"].min()) >= 2.0 * thr))
        checks.append((f"N: released hand {h} loaded <= {args.open:.2f} of the steps",
                       m(N, ck) <= args.open))
    checks.append((f"P: every recorded EMA load is >= 2x the threshold "
                   f"(worst {P_min:.1f} vs {2*thr:.1f} N.m)", P_min >= 2.0 * thr))
    checks.append((f"N: every recorded EMA load is <= threshold/2 "
                   f"(worst {N_max:.1f} vs {thr/2:.1f} N.m)", N_max <= 0.5 * thr))
    checks.append((f"H: the hovering left hand is inside B0's window while the "
                   f"force says no -- hnext_L peaks at {hnext_peak:.2f} "
                   f"while f_L,B0 peaks at {force_peak:.2f}",
                   hnext_peak >= args.hover_peak and force_peak <= args.open))
    checks.append((f"H: ... for at least 3 recorded steps "
                   f"(got {int(np.sum(H['cov_hover_L_on_steps']))})",
                   float(np.sum(H["cov_hover_L_on_steps"])) >= 3.0))
    checks.append((f"H: the hovering hand never carries load (max {H_L_max:.1f} "
                   f"< threshold {thr:.1f} N.m)", H_L_max < thr))
    checks.append((f"H: ... while the supporting right hand carries the body "
                   f"(min {H_R_min:.1f} after the EMA cold start >= 2x threshold)",
                   H_R_min >= 2.0 * thr))
    checks.append(("N: the robot actually fell (final torso z well below the bars)",
                   float(N["z"][-1]) < float(env0.bar_z) - 0.4))
    # The obs check is a *consistency* check, not a level check: the recorded
    # feature must be exactly 1-exp(-streak/10) of the recorded streak, and the
    # streak must have survived the whole recorded window.  (A level check such as
    # "> 0.9" is silently a check on --steps: 1-exp(-s/10) only passes 0.9 at s=23.)
    def feat(streak):
        return 1.0 - np.exp(-np.asarray(streak, dtype=np.float64) / 10.0)

    P_feat = feat(P["final_streak"])
    H_feat = feat(H["final_streak"])
    checks.append((f"obs plumbing: P's contact slots (state dims "
                   f"{env0.layout.contact}) == 1-exp(-streak/10) with a full streak "
                   f"(streak {P['final_streak']}, obs "
                   f"{np.round(P['obs_contact'], 3)} vs {np.round(P_feat, 3)})",
                   bool(np.all(P["final_streak"] >= args.steps)
                        and np.allclose(P["obs_contact"], P_feat, atol=1e-5))))
    checks.append((f"obs plumbing: H's hovering left slot is 0 (streak "
                   f"{H['final_streak'][0]}) while its supporting right slot is "
                   f"1-exp(-streak/10) (streak {H['final_streak'][1]}, obs "
                   f"{np.round(H['obs_contact'], 3)} vs {np.round(H_feat, 3)})",
                   bool(H["final_streak"][0] == 0.0
                        and H["final_streak"][1] >= 3.0
                        and H["obs_contact"][0] == 0.0
                        and abs(H["obs_contact"][1] - H_feat[1]) < 1e-5)))
    checks.append((f"obs plumbing: N/P2 contact slots and streaks are all 0 "
                   f"(obs {np.round(N['obs_contact'],3)} / "
                   f"{np.round(P2['obs_contact'],3)}, streaks "
                   f"{N['final_streak']} / {P2['final_streak']})",
                   bool(np.max(N["obs_contact"]) == 0.0
                        and np.max(P2["obs_contact"]) == 0.0
                        and np.max(N["final_streak"]) == 0.0
                        and np.max(P2["final_streak"]) == 0.0)))

    print("[contact] controls:")
    bad = 0
    for name, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
        bad += 0 if ok else 1
    if bad:
        print(f"FAIL {bad} control(s) did not hold")
        return 1
    print(f"OK  all {len(checks)} contact-sense controls hold "
          f"(F0={env0.contact_f0}, theta={env0.contact_theta}, "
          f"thresh={thr:.1f} N.m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
