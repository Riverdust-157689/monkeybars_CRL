"""Measure hand<->bar INTERPENETRATION for a policy rollout (native MuJoCo).

Why this exists
---------------
The physics is MJX-Warp, but the video is rendered by replaying the same qpos in
**native MuJoCo** -- and native MuJoCo is where the geometry can be inspected
directly.  A policy can satisfy every goal-space metric we have (contact flag,
`f` from `qfrc_constraint`, "hand inside the 5 cm window", legs of the reach)
while the hand mesh is actually INSIDE the bar capsule, because

  * the bar is a `CAPSULE` (r = 2.5 cm) and the hands are `MESH`/`BOX` geoms --
    exactly the pair that the Warp backend warns about with MULTICCD enabled:
    "the scene contains CCD pairs without multicontact support:
    [('CAPSULE','MESH'), ...]. At most 1 contact will be generated for these
    pairs", so a mesh sliding into a capsule cannot be resolved by several
    contacts the way a real hook would be;
  * MuJoCo's soft contacts legitimately allow a few mm of penetration (measured:
    the validated `hang` keyframe sits at -5.1 mm), so the useful question is not
    "is it negative" but "is it far beyond the ~5 mm that a real hook uses".

Calibration (printed by ``--reference``): the keyframe two-handed hang gives
min signed distance **-5.1 mm (left) / -5.0 mm (right)** to bar0.  Anything at
-1 cm or deeper is not a hook, it is the mesh cutting through the bar.

Usage::

    JAX_PLATFORMS=cpu .venv-warp/bin/python src/check_grasp_penetration.py \
        --ckpt runs/ckpt_lboth_2bar_6h/actor_latest.pkl --goal-bar 2 \
        --episodes 4 --steps 501

Exit code 1 if any hand penetrates deeper than ``--max-penetration`` (default
1 cm) in any recorded step.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "envs"), os.path.join(REPO, "third_party", "jaxgcrl")):
    if p not in sys.path:
        sys.path.insert(0, p)


def bar_and_hand_geoms(m):
    """({bar k: geom id}, {side: [geom ids of that wrist subtree]})."""
    import mujoco
    bars = {}
    for k in range(5):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"bar{k}")
        gs = list(np.flatnonzero(m.geom_bodyid == bid))
        if gs:
            bars[k] = int(gs[0])
    hands = {}
    for side in ("left", "right"):
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_roll_link")
        sub = []
        for b in range(m.nbody):
            p = b
            while p > 0 and p != root:
                p = m.body_parentid[p]
            if p == root or b == root:
                sub.append(b)
        hands[side] = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub]
    return bars, hands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", default=None, choices=["full", "full035", "mesh", "allprim"])
    ap.add_argument("--goal-bar", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=501)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-penetration", type=float, default=0.01,
                    help="fail if a hand goes deeper than this (m) into any bar")
    ap.add_argument("--near", type=float, default=0.20,
                    help="only test bars whose grasp-seat distance is below this")
    ap.add_argument("--reference", action="store_true",
                    help="also measure the `hang` keyframe (the calibration row)")
    # ---- physics knobs: does the SAME policy stop penetrating under a stricter
    # contact/solver setting?  (This is how to tell a policy exploit from a
    # simulator that simply cannot resolve the hand-bar contact.)
    ap.add_argument("--ls-iterations", type=int, default=None,
                    help="override opt.ls_iterations for the ROLLOUT (the scene pins 20)")
    ap.add_argument("--disable-multiccd", action="store_true",
                    help="set opt.disableflags |= MULTICCD for the rollout; the Warp "
                         "backend warns that (CAPSULE,MESH) pairs get at most 1 contact "
                         "while MULTICCD is on")
    ap.add_argument("--solref-timeconst", type=float, default=None,
                    help="override the hand/bar geoms' contact timeconst (default 0.02 s)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import jax
    import jax.numpy as jnp
    import mujoco
    import flax.linen as nn

    from envs.brachiation import create_brachiation
    from render_policy import load_actor
    from jaxgcrl.agents.crl.networks import Actor

    actor_params, cfg, ckpt_steps = load_actor(args.ckpt)
    scene = args.scene or cfg.get("scene", "full")
    impl = cfg.get("impl", "warp")
    variant = cfg.get("goal_variant", "full")
    goal_bar = None if args.goal_bar < 0 else args.goal_bar
    env = create_brachiation(impl=impl, scene=scene, n_frames=int(cfg.get("n_frames", 10)),
                             goal_bar=goal_bar, naconmax=4096, njmax=512,
                             goal_variant=variant, action_window=cfg.get("action_window", "reach"),
                             **({"ls_iterations": args.ls_iterations}
                                if args.ls_iterations else {}))
    # optional physics overrides for the ROLLOUT (diagnostic A/B on one policy)
    import mujoco as _mj
    if args.disable_multiccd or args.solref_timeconst:
        import mujoco.mjx as _mjx
        import mjx_backend as _mb
        if args.disable_multiccd:
            env.mj_model.opt.disableflags |= int(_mj.mjtDisableBit.mjDSBL_MULTICCD)
        if args.solref_timeconst:
            env.mj_model.geom_solref[:, 0] = float(args.solref_timeconst)
        env.mx_model = _mb.silence_warp_overflow(
            _mjx.put_model(env.mj_model, impl=env.impl), env.impl)
        print(f"[physics] rollout overrides: ls_iterations={env.mj_model.opt.ls_iterations} "
              f"multiccd={'OFF' if args.disable_multiccd else 'on'} "
              f"solref_tc={env.mj_model.geom_solref[0,0]}")
    actor = Actor(action_size=int(cfg.get("action_size", 14)),
                  network_width=int(cfg.get("h_dim", 256)),
                  network_depth=int(cfg.get("n_hidden", 17)),
                  skip_connections=int(cfg.get("skip_connections", 4)), use_relu=False)

    mj = mujoco.MjModel.from_xml_path(env.scene_xml)
    md = mujoco.MjData(mj)
    bars, hands = bar_and_hand_geoms(mj)
    fto = np.zeros(6)

    def penetration(qpos_row):
        """({side: (min_dist, bar_k)}, n_contacted) for one qpos."""
        md.qpos[:] = qpos_row
        mujoco.mj_forward(mj, md)
        out = {}
        for side, gs in hands.items():
            # nearest bar by the CLOSEST of this hand's geoms (mj_geomDistance
            # returns a signed distance: >0 = outside, <0 = penetrating)
            best = (1e9, -1)
            for k, bg in bars.items():
                d = min(mujoco.mj_geomDistance(mj, md, g, bg, args.near, fto) for g in gs)
                if d < best[0]:
                    best = (d, k)
            out[side] = best
        return out

    def report(tag, rows, loads=None):
        print(f"\n[{tag}]")
        worst = 0.0
        for e in range(len(rows[0])):
            col = [penetration(rows[t][e]) for t in range(len(rows))]
            for side in ("left", "right"):
                ds = np.array([c[side][0] for c in col])
                ks = [c[side][1] for c in col]
                deep = int(np.sum(ds < -args.max_penetration))
                print(f"  ep{e} {side:5s}: min {ds.min()*1000:+7.1f} mm | median "
                      f"{np.median(ds)*1000:+6.1f} mm | steps < -{args.max_penetration*1000:.0f} mm: "
                      f"{deep:3d}/{len(ds)} | bar in contact: "
                      f"{ {k: ks.count(k) for k in sorted(set(ks))} }")
                worst = min(worst, float(ds.min()))
            if loads is not None:
                L, R = loads[0][e], loads[1][e]
                print(f"        load EMA (N.m): L mean {L.mean():5.1f} max {L.max():5.1f} | "
                      f"R mean {R.mean():5.1f} max {R.max():5.1f}")
        print(f"  worst penetration overall: {worst*1000:+.1f} mm  "
              f"(threshold -{args.max_penetration*1000:.0f} mm)")
        return worst

    if args.reference:
        kid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_KEY, "hang")
        mujoco.mj_resetDataKeyframe(mj, md, kid)
        q = md.qpos.copy()
        mujoco.mj_forward(mj, md)
        r = penetration(q)
        print("[reference: `hang` keyframe, the validated two-handed hang]")
        for side in ("left", "right"):
            print(f"  {side:5s}: min signed dist {r[side][0]*1000:+.1f} mm to bar{r[side][1]}")

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    act = jax.jit(lambda p, o: nn.tanh(actor.apply(p, o)[0]))
    state = reset(jax.random.split(jax.random.PRNGKey(args.seed), args.episodes))
    rows, loads = [], []
    for t in range(args.steps + 1):
        rows.append(np.asarray(state.pipeline_state.qpos).copy())
        if t == args.steps:
            break
        state = step(state, act(actor_params, state.obs))
        loads.append((np.asarray(state.info["contact_ema"]).copy(),
                      np.asarray(state.info["contact_ema"]).copy()))
    L = np.stack([l[0] for l in loads])[:, :, 0].T if loads else None
    R = np.stack([l[0] for l in loads])[:, :, 1].T if loads else None
    worst = report(f"{args.ckpt} goal-bar={args.goal_bar} ({args.steps} steps x {args.episodes} eps)",
                   rows, (L, R) if loads else None)
    if worst < -args.max_penetration:
        print(f"FAIL a hand penetrates {-worst*1000:.1f} mm into a bar "
              f"(threshold {args.max_penetration*1000:.0f} mm)")
        return 1
    print("OK  no hand penetrates deeper than the threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
