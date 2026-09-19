"""Smoke test for the Brachiation env adapter (plan A).

Verifies, without any training:
  1. construction, obs/action sizes, `state_dim` / `goal_indices` contract
  2. `jax.jit(reset)` and `jax.jit(step)` compile and run
  3. JaxGCRL's own wrappers accept the env (`TrajectoryIdWrapper` + `envs.training.wrap`)
  4. a random-action rollout: obs shape, metrics appear, `done` triggers on a fall
  5. zero-action rollout from the keyframe keeps hanging (sanity vs M1)

Usage::

    .venv-warp/bin/python src/smoke_env.py                 # warp, full mesh, 1 env
    .venv-warp/bin/python src/smoke_env.py --impl jax --scene allprim
    .venv-warp/bin/python src/smoke_env.py --rollout 200 --vmap 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "envs"))

from envs.brachiation import create_brachiation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="warp", choices=["jax", "warp"])
    ap.add_argument("--scene", default="full", choices=["full", "full035", "mesh", "allprim"])
    ap.add_argument("--rollout", type=int, default=100)
    ap.add_argument("--vmap", type=int, default=1)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    t0 = time.time()
    env = create_brachiation(impl=args.impl, scene=args.scene, n_frames=10)
    print(f"[1] env built in {time.time()-t0:.1f}s  backend={env.backend}")
    print(f"    action_size={env.action_size}  observation_size={env.observation_size}  "
          f"state_dim={env.state_dim}  len(goal_indices)={len(env.goal_indices)}")
    print(f"    n_bars={env.n_bars} bar_x={np.array(env.bar_x)} bar_z={env.bar_z:.3f} "
          f"spacing={env.bar_spacing:.2f}")
    g = np.array(env.goal_set)
    print(f"    goal set {g.shape}; goal[0]={np.round(g[0], 3)}")
    assert env.observation_size == env.state_dim + len(env.goal_indices)

    key = jax.random.PRNGKey(0)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    t0 = time.time()
    state = reset(key)
    jax.block_until_ready(state.obs)
    print(f"[2] jit(reset) OK {time.time()-t0:.1f}s  obs={state.obs.shape} "
          f"goal={np.round(np.array(state.info['goal']), 3)}")
    assert state.obs.shape == (env.observation_size,)

    a = jnp.zeros(env.action_size)
    t0 = time.time()
    state = step(state, a)
    jax.block_until_ready(state.obs)
    print(f"    jit(step) OK {time.time()-t0:.1f}s  reward={float(state.reward)} "
          f"done={float(state.done)}")
    print(f"    metrics keys: {sorted(state.metrics.keys())}")

    # --- JaxGCRL's wrappers accept it (plan A: brax as a pure type library) ---
    # NOTE: brax's EpisodeWrapper does `jax.vmap(self.env.reset)(rng)`, and
    # JaxGCRL does `env_keys = jax.random.split(key, num_envs)` — so the env is
    # *always* reset/stepped under a vmap over a batch of keys.  Test it that way.
    try:
        from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
        from brax import envs as brax_envs
        N = 4
        wrapped = TrajectoryIdWrapper(env)
        wrapped = brax_envs.training.wrap(wrapped, episode_length=200, action_repeat=1)
        keys = jax.random.split(jax.random.PRNGKey(0), N)
        s = jax.jit(wrapped.reset)(keys)
        s = jax.jit(wrapped.step)(s, jnp.zeros((N, env.action_size)))
        jax.block_until_ready(s.obs)
        print(f"[3] JaxGCRL TrajectoryIdWrapper + brax envs.training.wrap OK  "
              f"(vmap x{N}) obs={s.obs.shape} steps={np.array(s.info['steps'])} "
              f"traj_id={np.array(s.info['traj_id'])}")
    except ImportError as e:
        print(f"[3] skipped (jaxgcrl not importable in this venv: {e})")

    # --- rollouts ---------------------------------------------------------
    def rollout(env, n, act_fn, keys):
        st = jax.jit(jax.vmap(env.reset))(keys) if keys.ndim > 1 else jax.jit(env.reset)(keys)
        for i in range(n):
            a = act_fn(i, keys)
            st = jax.jit(jax.vmap(env.step))(st, a) if keys.ndim > 1 else jax.jit(env.step)(st, a)
        return st

    # Action convention: q_des = key_ctrl + scale * a[:12], c = (a[12:]+1)/2 with
    # c=1 = closed grasp (see docs/手部控制说明.md).  So the "do nothing but keep
    # holding" action is a = [0]*12 + [1, 1]; a = 0 means a HALF-OPEN hand.
    hold = jnp.concatenate([jnp.zeros(12), jnp.ones(2)])
    key_z = float(env.key_qpos[2])
    for tag, act in (("hold  (a=[0]*12,[1,1])", hold),
                     ("zero  (a=0 -> half-open grasp)", jnp.zeros(env.action_size))):
        st = reset(key)
        for _ in range(args.rollout):
            st = step(st, act)
        z = float(st.pipeline_state.qpos[2])
        fell = z < key_z - 0.30
        print(f"[4] {tag:32s} ({args.rollout} steps = "
              f"{args.rollout*env.n_frames*0.002:.1f} s): pelvis z {key_z:.4f} -> {z:.4f} "
              f"(drift {z-key_z:+.4f} m)  done={float(st.done)}  "
              f"bar_L={float(st.metrics['bar_L'])} bar_R={float(st.metrics['bar_R'])}  "
              f"{'FELL' if fell else 'HANG HELD'}")

    if args.vmap > 1:
        N = args.vmap
        print(f"[5] random-action vmap rollout, {N} envs x {args.rollout} steps")
        keys = jax.random.split(jax.random.PRNGKey(1), N)
        t0 = time.time()
        st = jax.jit(jax.vmap(env.reset))(keys)
        stepv = jax.jit(jax.vmap(env.step))
        for i in range(args.rollout):
            a = jax.random.uniform(jax.random.PRNGKey(100 + i), (N, env.action_size), minval=-1, maxval=1)
            st = stepv(st, a)
        jax.block_until_ready(st.obs)
        w = time.time() - t0
        print(f"    {N*args.rollout} env-steps in {w:.1f}s -> {N*args.rollout/w:.1f} env-steps/s "
              f"({args.impl}, CPU numbers only)")
        print(f"    done fraction={float(jnp.mean(st.done)):.2f}  "
              f"mean max_bar={float(jnp.mean(st.metrics['max_bar'])):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
