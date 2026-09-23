"""Record a trained policy on the monkey bars and dump per-step diagnostics.

The policy is the actor of a CRL run: ``obs = [state(141) | goal(10)]`` -> a
squashed Gaussian, so the *deterministic* rollout uses ``a = tanh(mean(obs))``
exactly like JaxGCRL's ``deterministic_actor_step`` (which is what the evaluator
reports).

Input is one of the checkpoints written by ``src/train.py`` (``actor_*.pkl`` /
``actor_latest.pkl``) or JaxGCRL's own ``(alpha, actor, critic)`` tuple.  A run
that finished *without* ``--checkpoint-dir`` has no such file and cannot be
rendered -- that is the whole reason ``--save-every`` exists.

Usage (on the GPU machine)::

    .venv-warp/bin/python src/render_policy.py --ckpt ckpt/brach_C_gmin1/actor_latest.pkl
    .venv-warp/bin/python src/render_policy.py --ckpt ... --episodes 4 --steps 501
    MUJOCO_GL=egl .venv-warp/bin/python src/render_policy.py --ckpt ...   # headless fallback

Outputs (under ``runs/render/<name>/``):
  * ``frames/step_%04d.png``  -- always written (needs no ffmpeg)
  * ``<name>.mp4``            -- if `imageio` is installed
  * ``<name>.gif``            -- pillow fallback
  * ``trace.csv``             -- per-step qpos / grasp / goal-distance diagnostics
  * a printed summary (bars reached, falls, contact switches, distance drift)

Rendering uses the *native* MuJoCo model (the same XML) and replays the qpos that
MJX-Warp produced, so the video is the physics we actually trained on.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "envs"),
        # the vendored upstream JaxGCRL (third_party/jaxgcrl) -- it is NOT a pip
        # dependency any more, so it must be on sys.path for `import jaxgcrl`
        os.path.join(REPO, "third_party", "jaxgcrl"),
          ):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_actor(path: str):
    """Return ``(actor_params, config_dict, steps)`` from either ckpt format."""
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    if isinstance(blob, dict) and "actor" in blob:
        return blob["actor"], dict(blob.get("config", {})), int(blob.get("steps", -1))
    if isinstance(blob, (tuple, list)) and len(blob) == 3:
        # JaxGCRL's save format: (alpha_params, actor_params, critic_params)
        return blob[1], {}, -1
    raise SystemExit(f"{path}: unrecognised checkpoint (type {type(blob)}, "
                     f"keys {list(blob) if isinstance(blob, dict) else '-'})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="actor_*.pkl / final / actor tuple")
    ap.add_argument("--scene", default=None, choices=["full", "full035", "mesh", "allprim"])
    ap.add_argument("--impl", default=None, choices=["jax", "warp"])
    ap.add_argument("--goal-bar", type=int, default=4,
                    help="instruction goal for the rollout (default B4 = the eval goal); -1 = random")
    ap.add_argument("--episodes", type=int, default=1, help="parallel episodes (shown side by side)")
    ap.add_argument("--steps", type=int, default=501, help="env steps per episode (0.02 s each)")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of tanh(mean) (the training behaviour)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="output name (default: ckpt file stem)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=50, help="1 frame per env step = 50 fps")
    ap.add_argument("--azimuth", type=float, default=70.0)
    ap.add_argument("--elevation", type=float, default=-8.0)
    ap.add_argument("--distance", type=float, default=3.4)
    ap.add_argument("--lookat-x", type=float, default=0.8,
                    help="fixed camera target along x (0.8 = middle of the 5 bars, "
                         "spacing 0.4 m); the bars are welded to the world, so a "
                         "camera that translates makes them look like they move")
    ap.add_argument("--follow", type=int, default=0,
                    help="1 = camera tracks the torso along x (bars will appear to slide); "
                         "0 = fixed camera (default)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")   # must precede `import mujoco`
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import mujoco

    import mjx_backend as mb
    from envs.brachiation import create_brachiation
    from jaxgcrl.agents.crl.networks import Actor

    actor_params, cfg, ckpt_steps = load_actor(args.ckpt)
    scene = args.scene or cfg.get("scene", "full")
    impl = args.impl or cfg.get("impl", "warp")
    h_dim = int(cfg.get("h_dim", 256))
    n_hidden = int(cfg.get("n_hidden", 17))
    skip = int(cfg.get("skip_connections", 4))
    action_size = int(cfg.get("action_size", 14))
    n_frames = int(cfg.get("n_frames", 10))

    if impl == "warp":
        mb.enable_warp_compat()
    goal_bar = None if args.goal_bar < 0 else args.goal_bar
    # the trace's d_* columns / closest-hand summary are relative to this bar
    k_goal = int(goal_bar) if goal_bar is not None else 1
    # must match training, otherwise the actor's input width is wrong
    variant = cfg.get("goal_variant") or ("position" if cfg.get("goal_position_only") else "full")
    window = cfg.get("action_window", "reach")
    contact = {k: float(cfg[k]) for k in ("contact_f0", "contact_theta", "contact_ema")
               if k in cfg}
    env = create_brachiation(impl=impl, scene=scene, n_frames=n_frames,
                             goal_bar=goal_bar, naconmax=4096, njmax=512,
                             goal_variant=variant, action_window=window, **contact)
    # upstream CRL never forwards use_ln to the Actor -> instantiate it the same way
    actor = Actor(action_size=action_size, network_width=h_dim, network_depth=n_hidden,
                  skip_connections=skip, use_relu=False)

    E, T = args.episodes, args.steps
    ctxt = (f"contact_thresh={env.contact_thresh:.1f} N.m (f0={env.contact_f0:g}, "
            f"theta={env.contact_theta:g}, ema={env.contact_ema:g}) "
            if "contact" in variant else "")
    print(f"[render] ckpt={args.ckpt} steps={ckpt_steps} git={cfg.get('git_commit', '?')} "
          f"scene={scene} impl={impl} "
          f"W={h_dim} D={n_hidden} skip={skip} n_frames={n_frames} "
          f"goal_bar={goal_bar} goal_variant={variant} action_window={window} "
          f"obs={env.observation_size} episodes={E} steps={T} {ctxt}"
          f"actor={'stochastic' if args.stochastic else 'deterministic (eval mode)'}",
          flush=True)

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    grasp = jax.jit(jax.vmap(env._grasp))
    # ONE entry point for the goal readout, whatever the variant: the old code
    # hand-assembled `_achieved_goal`'s per-variant arguments here and got them
    # wrong twice (a variant whose history argument was dropped silently reports a
    # goal distance computed with that dim = 0).  See `_achieved_goal_info`.
    achieved = jax.jit(jax.vmap(env._achieved_goal_info))
    state = reset(jax.random.split(jax.random.PRNGKey(args.seed), E))

    def act(params, obs, key):
        mean, log_std = actor.apply(params, obs)
        if not args.stochastic:
            return nn.tanh(mean)
        return nn.tanh(mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape))

    act_jit = jax.jit(act)
    key = jax.random.PRNGKey(args.seed + 1)

    qpos = np.zeros((T + 1, E, env.mj_model.nq), dtype=np.float32)
    rec = {k: np.zeros((T + 1, E), dtype=np.float32)
           for k in ("dist", "dist_pos", "c_L", "c_R", "bar_L", "bar_R", "done", "switched",
                     "d_LB1", "d_RB1")}
    # "alive at the START of this step", i.e. the live-phase mask (see below).
    # It must be recorded per step: `ever_done` is only the CURRENT accumulator
    # (shape (E,)), so masking with it afterwards silently collapses to the final
    # all-fallen vector.
    rec["live"] = np.zeros((T + 1, E), dtype=np.float32)
    # 9.72/9.74: "is the hand on the instructed bar HOVERING (in the window, no
    # load) or really GRIPPING (carrying load)?" is worth recording for EVERY
    # variant, not just the contact family: the env computes the load streaks for
    # all of them (`info["contact_ema"]`, `cov_load_*`), and asking the same
    # question of the distance-based baselines is how we tell whether THEIR
    # failure at the second bar is a missing-load problem or something else.
    uses_contact = True
    rec.update({k: np.zeros((T + 1, E), dtype=np.float32)
                for k in ("f_L", "f_R", "near_L", "near_R", "load_L", "load_R",
                          "hover_L", "hover_R")})

    def snapshot(i, s, ever_done):
        qpos[i] = np.asarray(s.pipeline_state.qpos)
        d, w = (np.asarray(x) for x in grasp(s.pipeline_state))
        dmin = d.min(axis=-1)
        bar = np.where(dmin < 0.05, w.argmax(axis=-1), -1)
        # measure the goal distance ourselves: reset() seeds metrics with zeros.
        # `_achieved_goal_info` reads every history-dependent arg (hold, h_dual,
        # h_park, k_goal, hnext, contact) out of `state.info` -- vmap over a dict
        # pytree works, and it cannot drift from what `step` computes.
        diff = (np.asarray(achieved(s.pipeline_state, s.info))
                - np.asarray(s.info["goal"]))
        rec["dist"][i] = np.linalg.norm(diff, axis=-1)
        rec["dist_pos"][i] = np.linalg.norm(diff[:, :8], axis=-1)   # position-only view (drop c_L,c_R)
        rec["c_L"][i] = np.exp(-(dmin[:, 0] / 0.04) ** 2)
        rec["c_R"][i] = np.exp(-(dmin[:, 1] / 0.04) ** 2)
        rec["bar_L"][i], rec["bar_R"][i] = bar[:, 0], bar[:, 1]
        # per-hand distance to the INSTRUCTION bar (9.69: it used to be hardcoded to B1,
        # which silently reported B1 distances when rendering with --goal-bar 2+)
        if d.shape[-1] > 1:
            _k = k_goal
            rec["d_LB1"][i], rec["d_RB1"][i] = d[:, 0, _k], d[:, 1, _k]
            if uses_contact:
                near = d[:, :, _k] < 0.05                       # (E, 2) distance gate
                load = np.asarray(s.info["contact_ema"])        # (E, 2) EMA finger load
                f = np.asarray(env._hold_feature(s.info["contact_streak"]))
                rec["near_L"][i], rec["near_R"][i] = near[:, 0], near[:, 1]
                rec["load_L"][i], rec["load_R"][i] = load[:, 0], load[:, 1]
                rec["f_L"][i], rec["f_R"][i] = f[:, 0], f[:, 1]
                loaded = load > env.contact_thresh
                rec["hover_L"][i], rec["hover_R"][i] = near[:, 0] & ~loaded[:, 0], \
                    near[:, 1] & ~loaded[:, 1]
        rec["switched"][i] = np.asarray(s.metrics["hand_switches"])
        rec["done"][i] = np.asarray(s.done)
        rec["live"][i] = ~ever_done          # alive BEFORE this step's `done`
        return ever_done | (np.asarray(s.done) > 0)

    ever_done = snapshot(0, state, np.zeros(E, dtype=bool))
    for t in range(T):
        key, k = jax.random.split(key)
        state = step(state, act_jit(actor_params, state.obs, k))
        ever_done = snapshot(t + 1, state, ever_done)

    # ---------------------------------------------------------------- summary
    max_bar = np.clip(np.maximum(rec["bar_L"], rec["bar_R"]), 0, None).max(axis=1)
    run_max = np.maximum.accumulate(max_bar, axis=0)
    fell_at = [int(np.argmax(rec["done"][:, e] > 0)) if (rec["done"][:, e] > 0).any() else -1
               for e in range(E)]
    # ---- live-phase mask ---------------------------------------------------
    # Every mean below must be taken over the steps BEFORE the fall: after `done`
    # the env keeps stepping (there is no auto-reset in this loop), so an
    # all-episode mean of a per-step quantity is dominated by hundreds of
    # free-fall steps and reads ~0 no matter what the policy did.  (This is the
    # same mistake as counting the free-fall tail as behaviour; see
    # docs/M2_JaxGCRL接入记录.md 9.61/9.62.)
    live = rec["live"] > 0
    n_live = live.sum(axis=0)
    print(f"[result] live steps      = {n_live.tolist()} of {T + 1} "
          f"(everything below is measured on these only)")

    def mask(x):
        return x[live]

    print(f"[result] episode max bar = {run_max[-1].astype(int).tolist()}   (0 = never left B0)")
    print(f"[result] fell at step    = {fell_at}   (-1 = survived the episode)")
    print(f"[result] live steps      = {n_live.tolist()} of {T + 1} "
          f"(everything below is measured on these only)")
    print(f"[result] goal dist       = {rec['dist'][0].mean():.3f} -> best(live) "
          f"{rec['dist'].min(axis=0).mean():.3f} ({len(env.goal_indices)}-dim)   "
          f"[position-only 8-dim: {rec['dist_pos'][0].mean():.3f} -> "
          f"{rec['dist_pos'].min(axis=0).mean():.3f}]")
    print(f"[result] torso x         = {qpos[0, :, 0].mean():+.3f} -> {qpos[-1, :, 0].mean():+.3f} m "
          f"(bar spacing {env.bar_spacing:.3f} m)")
    print(f"[result] torso z         = {qpos[0, :, 2].mean():.3f} -> {qpos[-1, :, 2].mean():.3f} m")
    print(f"[result] hand switches   = {rec['switched'].sum(axis=0).astype(int).tolist()} per episode")
    if "d_LB1" in rec:
        L1, R1 = rec["d_LB1"].min(axis=0), rec["d_RB1"].min(axis=0)
        who = ["left" if l < r else "right" for l, r in zip(L1, R1)]
        print(f"[result] closest hand to B{k_goal} per episode = {who}  "
              f"(L {np.round(L1,3).tolist()} vs R {np.round(R1,3).tolist()}, threshold 0.05)")
    print(f"[result] grasping (soft, live) = L {mask(rec['c_L']).mean():.2f}  "
          f"R {mask(rec['c_R']).mean():.2f} "
          f"| best-step L {mask(rec['c_L']).max():.2f} R {mask(rec['c_R']).max():.2f} "
          f"(1 = closed on the bar)")
    if uses_contact:
        # The decisive readout for the contact family: `f` is the sustained REAL
        # grip on the INSTRUCTION bar, `hover` is "inside its window but carrying
        # no load" -- the state the distance criterion (hnext) scores as a grip.
        def frac(x):
            return float(np.mean(x > 0.5)) * 100.0
        print(f"[result] REAL grip on B{k_goal} (f, live): "
              f"L mean {mask(rec['f_L']).mean():.2f} (f>0.5 in {frac(mask(rec['f_L'])):.0f}% of live steps)  "
              f"R mean {mask(rec['f_R']).mean():.2f} (f>0.5 in {frac(mask(rec['f_R'])):.0f}%)")
        print(f"[result] HOVER on B{k_goal} (live; in the 5 cm window, NOT loaded): "
              f"L {100*mask(rec['hover_L']).mean():.0f}%  R {100*mask(rec['hover_R']).mean():.0f}%  "
              f"| in-window at all: L {100*mask(rec['near_L']).mean():.0f}%  "
              f"R {100*mask(rec['near_R']).mean():.0f}%")
        print(f"[result] finger load EMA (live, N.m): L mean {mask(rec['load_L']).mean():.1f} "
              f"(max {mask(rec['load_L']).max():.1f})  R mean {mask(rec['load_R']).mean():.1f} "
              f"(max {mask(rec['load_R']).max():.1f})  [threshold {env.contact_thresh:.1f}]")

    name = args.name or os.path.splitext(os.path.basename(args.ckpt))[0]
    outdir = os.path.join(REPO, "runs", "render", name)
    frames_dir = os.path.join(outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    with open(os.path.join(outdir, "trace.csv"), "w") as fh:
        cols = ("step,episode,qpos_x,qpos_y,qpos_z,dist,dist_pos,c_L,c_R,bar_L,bar_R,"
                "d_LB1,d_RB1,done,hand_switch")
        if uses_contact:
            cols += ",f_L,f_R,near_L,near_R,load_L,load_R,hover_L,hover_R"
        fh.write(cols + "\n")
        for t in range(T + 1):
            for e in range(E):
                row = (f"{t},{e},{qpos[t,e,0]:.6f},{qpos[t,e,1]:.6f},{qpos[t,e,2]:.6f},"
                       f"{rec['dist'][t,e]:.6f},{rec['dist_pos'][t,e]:.6f},"
                       f"{rec['c_L'][t,e]:.6f},{rec['c_R'][t,e]:.6f},"
                       f"{rec['bar_L'][t,e]:.0f},{rec['bar_R'][t,e]:.0f},"
                       f"{rec['d_LB1'][t,e]:.6f},{rec['d_RB1'][t,e]:.6f},"
                       f"{rec['done'][t,e]:.0f},{rec['switched'][t,e]:.0f}")
                if uses_contact:
                    row += (f",{rec['f_L'][t,e]:.4f},{rec['f_R'][t,e]:.4f},"
                            f"{rec['near_L'][t,e]:.0f},{rec['near_R'][t,e]:.0f},"
                            f"{rec['load_L'][t,e]:.2f},{rec['load_R'][t,e]:.2f},"
                            f"{rec['hover_L'][t,e]:.0f},{rec['hover_R'][t,e]:.0f}")
                fh.write(row + "\n")

    # ------------------------------------------------------------- rendering
    mj_model = mujoco.MjModel.from_xml_path(env.scene_xml)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance

    bar_z = float(env.bar_z)
    every = max(1, int(round(50.0 / args.fps)))
    frames = []
    for t in range(0, T + 1, every):
        row = []
        for e in range(E):
            mj_data.qpos[:] = qpos[t, e]
            mujoco.mj_forward(mj_model, mj_data)
            look_x = float(qpos[t, e, 0]) if args.follow else args.lookat_x
            cam.lookat[:] = [look_x, 0.0, bar_z - 0.45]
            renderer.update_scene(mj_data, cam)
            row.append(renderer.render())
        frames.append(row[0] if E == 1 else np.concatenate(row, axis=1))
    print(f"[render] {len(frames)} frames at {args.fps} fps "
          f"({(T + 1) * 0.02:.1f} s of physics per episode)")

    try:
        from PIL import Image
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(frames_dir, f"step_{i:04d}.png"))
        print(f"[png]    {frames_dir}/step_%04d.png")
    except Exception as exc:
        print(f"[warn] PNG frames failed: {exc}")

    written = None
    try:
        import imageio.v2 as imageio
        written = os.path.join(outdir, f"{name}.mp4")
        imageio.mimsave(written, frames, fps=args.fps, macro_block_size=1)
    except Exception as exc:
        print(f"[warn] mp4 failed ({type(exc).__name__}: {exc}); falling back to gif")
        try:
            from PIL import Image
            written = os.path.join(outdir, f"{name}.gif")
            Image.fromarray(frames[0]).save(
                written, save_all=True,
                append_images=[Image.fromarray(f) for f in frames[1:]],
                duration=int(1000 / args.fps), loop=0)
        except Exception as exc2:
            print(f"[warn] gif failed too: {exc2}")
            written = None
    if written:
        print(f"[video]  {written}")
    else:
        print(f"[hint]  PNG frames are in {frames_dir}; make a video with:\n"
              f"        ffmpeg -framerate {args.fps} -i {frames_dir}/step_%04d.png "
              f"-pix_fmt yuv420p {os.path.join(outdir, name)}.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
