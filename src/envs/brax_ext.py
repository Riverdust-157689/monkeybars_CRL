"""brax env-wrapper extensions needed to host an MJX environment (plan A).

JaxGCRL's CRL unconditionally calls ``brax.envs.training.wrap``.  That stack is
physics-agnostic (it only manipulates the ``State`` pytree), **except** for one
bug when ``pipeline_state`` is an ``mjx.Data``:

    brax/envs/wrappers/training.py :: AutoResetWrapper.where_done
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))

``jax.tree.map`` walks *every* leaf of the MJX data, and MJX data contains
legitimately empty leaves (e.g. ``act`` has size 0 for a model with no activation
dynamics).  For such a leaf ``x.shape[0] == 0`` and the reshape fails with

    TypeError: cannot reshape array of shape (4,) (size 4) into shape [0] (size 0)

Fix: skip empty (and scalar) leaves — there is nothing per-environment to swap
for a zero-length buffer.  We re-implement only ``AutoResetWrapper`` and reuse
brax's ``EpisodeWrapper`` / ``VmapWrapper`` / ``EvalWrapper`` unchanged.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jp
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers import training as brax_training


class MjxAutoResetWrapper(Wrapper):
    """``brax.envs.wrappers.training.AutoResetWrapper`` that tolerates empty leaves."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            state.info.update(steps=jp.where(state.done, jp.zeros_like(steps), steps))
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            """Swap x (the first state) for y only on leaves that are per-env.

            Three kinds of leaves must be left alone:
              * empty buffers (``mjx.Data.act`` has size 0);
              * scalars;
              * MJX-Warp ``_impl`` scratch buffers, which are *not* batched
                (Warp keeps them in its own FFI storage, so their leading dim is
                ``naconmax``, not ``num_envs``).  They are recomputed by
                ``mjx.step`` from qpos/qvel, so not restoring them is harmless.
            """
            done = state.done
            if (done.shape and getattr(x, "ndim", 0) >= 1
                    and x.shape[0] == done.shape[0]):
                done_r = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
                return jp.where(done_r, x, y)
            return y

        pipeline_state = jax.tree.map(
            where_done, state.info["first_pipeline_state"], state.pipeline_state
        )
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


def wrap(env: Env, episode_length: int = 1000, action_repeat: int = 1,
         randomization_fn=None, **kwargs) -> Env:
    """Drop-in replacement for ``brax.envs.training.wrap`` (same order)."""
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = brax_training.VmapWrapper(env)
    env = MjxAutoResetWrapper(env)
    if randomization_fn is not None:
        env = brax_training.DomainRandomizationVmapWrapper(env, randomization_fn)
    return env
