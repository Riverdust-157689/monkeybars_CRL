"""Does a randomly perturbed reset always keep the hang? -- the reset-tolerance test.

The env reset is ``hang keyframe + noise + a short settle`` (see
``envs/brachiation.py::Brachiation.reset``): the whole free base gets uniform
``+-reset_noise_root`` m on x/y/z, every leg joint ``+-reset_noise_leg`` rad, every
arm/hand joint ``+-reset_noise_arm`` rad, and all velocities ``reset_noise_vel *
N(0,1)``; then the sim is stepped ``n_frames`` times with the keyframe ctrl so the
perturbed grasp can re-seat.

This script answers, quantitatively:

  1. right after reset, do BOTH hands still find bar 0?  (must be 100 % -- otherwise
     the "random initial state" is not a hang at all)
  2. do both hands stay on bar 0 for ``--steps`` env steps (default 2 s) of the HOLD
     action, and does the robot ever drop below the fall threshold in that time?
  3. how does that degrade with the noise scale -- i.e. what is the safety margin
     around the default 1x noise?
  4. did MJX-Warp drop any contacts?  (``ovf`` column)  Its ``naconmax`` is a
     **global** contact budget, so a batch-sized run with a small ``naconmax``
     silently loses contacts -- that would make "still hanging" meaningless.

Usage (on the GPU machine -- this takes a couple of minutes there, because each
scale recompiles the reset/step jits)::

    .venv-warp/bin/python src/check_reset.py                       # sweep, warp, full scene
    .venv-warp/bin/python src/check_reset.py --n 256 --steps 100
    .venv-warp/bin/python src/check_reset.py --scales 0 1 2        # quick, 3 scales
    .venv-warp/bin/python src/check_reset.py --impl jax --scene allprim   # CPU-only fallback
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mjx_backend as mb  # noqa: E402


def bars_from_grasp(d, w, threshold):
    """Which bar each hand grasps (or -1), from ``_grasp`` outputs.

    ``d``/``w`` are batched ``(N, 2, n_bars)``: ``_grasp`` must be vmapped over the
    batched pipeline state (in a vmapped state ``xmat`` is ``(N, nbody, 3, 3)``).
    """
    d, w = np.asarray(d), np.asarray(w)
    return np.where(d.min(axis=-1) < threshold, w.argmax(axis=-1), -1)


# mujoco_warp._src.types.OverflowType
OVF_NAMES = {
    1: "NEFC", 2: "NJMAX_NNZ", 4: "BROADPHASE", 8: "NARROWPHASE", 16: "CCD",
    32: "HFIELD", 64: "CONTACT_MATCH", 128: "NVMAX", 256: "EPA_HORIZON",
    512: "ITERATIONS", 1024: "LS_ITERATIONS",
}
# Bits that mean "the physics of this step is not trustworthy" (loss of contacts
# or constraint rows).  ITERATIONS / LS_ITERATIONS only mean "the solver stopped
# early", i.e. a less converged but still well-posed step.
OVF_FATAL_MASK = 511


def decode_ovf(v: int) -> str:
    if v < 0:
        return "n/a"
    if v == 0:
        return "0 (clean)"
    return "|".join(name for bit, name in OVF_NAMES.items() if v & bit) or str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="warp", choices=["jax", "warp"])
    ap.add_argument("--scene", default="full", choices=["full", "full035", "mesh", "allprim"])
    ap.add_argument("--n", type=int, default=128, help="parallel resets per scale")
    ap.add_argument("--steps", type=int, default=100, help="env steps held (100 x 0.02 s = 2 s)")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
                    help="multipliers on the default reset noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--naconmax-per-world", type=int, default=128,
                    help="MJX-Warp contact budget per world; the env gets "
                         "per_world*n (Warp's naconmax is a GLOBAL budget, and a "
                         "too-small value makes it silently drop contacts)")
    ap.add_argument("--njmax", type=int, default=512)
    ap.add_argument("--action-window", default="reach", choices=["m1", "reach"])
    ap.add_argument("--ls-iterations", type=int, default=0,
                    help="0 = keep the scene XML value (20); 50 = MuJoCo's default, "
                         "an A/B test for solver convergence")
    args = ap.parse_args()

    if args.impl == "warp":
        mb.enable_warp_compat()
    import jax
    import jax.numpy as jnp
    from envs.brachiation import FALL_DEPTH, GRASP_THRESH, create_brachiation

    # 1x reference noise == the env defaults
    ls_kw = {"ls_iterations": args.ls_iterations} if args.ls_iterations else {}
    ref = create_brachiation(impl=args.impl, scene=args.scene, n_frames=10, **ls_kw)
    nr, na, nl, nv = (float(x) for x in ref._reset_cfg)
    bar_z = float(ref.bar_z)
    print(f"scene={args.scene} impl={args.impl} n_frames={ref.n_frames} "
          f"bar_z={bar_z:.4f} fall_z={bar_z - FALL_DEPTH:.4f} "
          f"ls_iterations={ref.ls_iterations}")
    print(f"1x noise: root +-{nr*1000:.1f} mm, arm +-{na:.3f} rad, "
          f"leg +-{nl:.3f} rad, |qvel| ~ {nv:.3f}")
    print(f"{'scale':>6} {'grasp@reset':>11} {'held':>7} {'alive':>7} "
          f"{'z_min':>9} {'dz_mean':>8} {'fell':>5} {'ovf':>5} {'wall':>7}")

    verdict = {}
    for scale in args.scales:
        env = create_brachiation(impl=args.impl, scene=args.scene, n_frames=10, goal_bar=0,
                                 njmax=args.njmax, action_window=args.action_window,
                                 naconmax=max(1024, args.naconmax_per_world * args.n),
                                 reset_noise_root=nr * scale, reset_noise_arm=na * scale,
                                 reset_noise_leg=nl * scale, reset_noise_vel=nv * scale,
                                 **ls_kw)
        N = args.n
        keys = jax.random.split(jax.random.PRNGKey(args.seed), N)
        t0 = time.time()
        reset = jax.jit(jax.vmap(env.reset))
        s = reset(keys)
        # _grasp works on ONE unbatched mjx.Data, so vmap it over the batch
        grasp = jax.jit(jax.vmap(env._grasp))
        b0 = bars_from_grasp(*grasp(s.pipeline_state), GRASP_THRESH)
        both0 = (b0[:, 0] == 0) & (b0[:, 1] == 0)
        z0 = np.array(s.pipeline_state.qpos[:, 2])
        ovf = mb.overflow_bits(s.pipeline_state)
        ovf_max = int(np.bitwise_or.reduce(ovf)) if ovf is not None else -1

        hold = jnp.concatenate([jnp.zeros((N, 12)), jnp.ones((N, 2))], axis=1)
        step = jax.jit(jax.vmap(env.step))
        zmin, held, z = z0.copy(), both0.copy(), z0
        for _ in range(args.steps):
            s = step(s, hold)
            z = np.array(s.pipeline_state.qpos[:, 2])
            zmin = np.minimum(zmin, z)
            b = bars_from_grasp(*grasp(s.pipeline_state), GRASP_THRESH)
            held &= (b[:, 0] == 0) & (b[:, 1] == 0)
            o = mb.overflow_bits(s.pipeline_state)
            if o is not None:
                ovf_max |= int(np.bitwise_or.reduce(o))
        fell = zmin < bar_z - FALL_DEPTH
        alive = held & ~fell

        print(f"{scale:6.1f} {both0.mean()*100:10.1f}% {held.mean()*100:6.1f}% "
              f"{alive.mean()*100:6.1f}% {zmin.min():9.4f} {(z-z0).mean():8.4f} "
              f"{fell.sum():5d} {ovf_max:5d} {time.time()-t0:6.1f}s", flush=True)
        verdict[scale] = (both0.mean(), held.mean(), alive.mean(), ovf_max)

    print(f"\ngrasp@reset : both hands still on bar 0 immediately after reset (must be 100 %)")
    print(f"held        : both hands on bar 0 at EVERY step during the {args.steps*0.02:.1f} s hold")
    print(f"alive       : held AND never below the fall threshold z={bar_z-FALL_DEPTH:.3f}")
    print("              NB: 'grasp' is a GEOMETRIC proxy (hand centre = wrist pose +")
    print("              keyframe-calibrated seat offset), so large perturbations can")
    print("              break it while the fingers still touch -- 'fell' is the hard")
    print("              criterion and should be trusted first.")
    print("dz_mean     : mean root-z drift over the hold (M1 saw ~5 mm per 60 s);")
    print("              a fell robot free-falls 0.5*g*t^2 ~ 19.6 m in 2 s (no ground).")
    print("ovf         : OR of the Warp overflow bitmask over all steps/worlds. "
          "0 = clean. CONTACT/CONSTRAINT bits (NEFC|NJMAX_NNZ|BROADPHASE|NARROWPHASE|"
          "CCD|HFIELD|NVMAX|EPA_HORIZON) invalidate the row; "
          "ITERATIONS / LS_ITERATIONS only mean the solver stopped early.")
    print("HOLD action : a[0:12]=0 (arms at keyframe), a[12:14]=+1 (grippers closed)")
    for scale, (g0, h, a, ov) in verdict.items():
        fatal = ov & OVF_FATAL_MASK
        tag = "SAFE " if (g0 == 1.0 and h == 1.0 and (ov < 0 or not fatal)) else "CHECK"
        print(f"  {tag} scale {scale:>4.1f}x : grasp@reset {g0*100:.1f}%  held {h*100:.1f}%  "
              f"alive {a*100:.1f}%  overflow={decode_ovf(ov)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
