"""Training entry point: JaxGCRL's CRL learner on our MJX brachiation env (plan A).

Environment:  ``src/envs/brachiation.py`` (physics via ``mjx.step``; brax used only
for the ``State``/``Wrapper`` types).
Losses: **upstream JaxGCRL, unmodified.**  ``--preset`` only picks among the
combinations upstream already supports (``energy_fn`` x ``contrastive_loss_fn``);
the default is the Scaling-CRL recipe, i.e. **L2 score + forward InfoNCE**
(``--preset C_l2_infonce``).  ``Losses`` are not re-implemented here.

Usage::

    .venv-warp/bin/python src/train.py --preset C_l2_infonce --smoke
    .venv-warp/bin/python src/train.py --preset C_l2_infonce --steps 5000000 \
        --num-envs 128 --impl warp --scene full --exp-name brach_C

Outputs ``runs/<exp>/progress.csv`` (per-eval metrics) and ``runs/<exp>/args.json``.

Constraint (JaxGCRL checks it): ``num_envs * (episode_length - 1) % batch_size == 0``.
With the defaults (501, 512) that means ``num_envs`` must be a multiple of 128.
Evaluation pins the instruction goal to ``--eval-goal-bar`` (default B4) so that
``eval/episode_success`` is comparable across runs.

⚠️ Replay-buffer size is **per env**: JaxGCRL's queue is allocated with shape
``(max_replay_size, num_envs, data_size)``, so the device memory is
``max_replay_size x num_envs x data_size x 4`` bytes.  With our transition
(obs 151 + action 14 + 4 scalars = 169 floats = 676 B) and 128 envs,
``--max-replay-size 100000`` asks for **8.06 GiB** and dies on an 8 GB card.
``--buffer-gb`` (default 2.0) caps it: the effective value is clamped and printed.
Useful points on a 8 GB card at 128 envs:

| ``--max-replay-size`` | transitions | buffer | note |
|---|---|---|---|
| 10000 | 1.28 M | 0.81 GiB | comfortable |
| **20000 (default)** | 2.56 M | 1.61 GiB | still comfortable |
| 40000 | 5.12 M | 3.22 GiB | = upstream's *total* (512 envs x 10000); pair it with ``XLA_PYTHON_CLIENT_MEM_FRACTION=.90`` |

JAX preallocates 75% of the GPU by default, so ~2 GiB of buffer plus envs, nets
and XLA executables all fit inside that pool without touching the fraction env var.

⚠️ **MJX-Warp allocates OUTSIDE that pool.**  Warp's buffers (`mjx.Data` and the
per-step collision contexts) come from Warp's own allocator, and its `naconmax` is
a **global** budget, so every env must scale it by *its own* world count.  On an
8 GB card the ladder is:

| symptom | knob |
|---|---|
| `Warp CUDA error 2: out of memory` / `Failed to allocate ... on device` | lower `--xla-mem-fraction` (0.70 -> 0.65 -> 0.60) to leave Warp room |
| still short, or JAX OOM instead | `--max-replay-size 12000` (1.0 GiB), `--naconmax-per-world 64`, `--num-eval-envs 8` |
| JAX `RESOURCE_EXHAUSTED` | raise `--xla-mem-fraction` back towards 0.75 |

Default settings assume the 8 GB RTX 4060: `--num-eval-envs 16` (eval is the
biggest single allocation: a 501-step scan) and `--xla-mem-fraction 0.70`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

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

import mjx_backend as mb  # noqa: E402

# --------------------------------------------------------------------------- #
# Activation: Swish everywhere.  This is the paper's recipe (and upstream's
# default, ``use_relu=False``) -- the paper's ablation only *warns* against
# swapping Swish -> ReLU, so we keep it fixed rather than exposing a knob.
# --------------------------------------------------------------------------- #
USE_RELU = False

# --------------------------------------------------------------------------- #
# Loss presets: ONLY combinations that upstream JaxGCRL already supports.
# ``losses.py`` is untouched (no bias/scale, no re-implemented objectives).
# Default = Scaling-CRL recipe: L2 score + forward InfoNCE.
# --------------------------------------------------------------------------- #
PRESETS = {
    # Scaling CRL (paper) / our default: metric score + relative (InfoNCE) classification
    "C_l2_infonce": dict(energy_fn="l2", contrastive_loss_fn="fwd_infonce",
                         logsumexp_penalty_coeff=0.1),
    # same, dot-product score
    "A_dot_infonce": dict(energy_fn="dot", contrastive_loss_fn="fwd_infonce",
                          logsumexp_penalty_coeff=0.1),
    # JaxGCRL's own out-of-the-box combination (negative distance, not squared)
    "upstream_norm": dict(energy_fn="norm", contrastive_loss_fn="fwd_infonce",
                          logsumexp_penalty_coeff=0.1),
    # symmetric InfoNCE (both directions)
    "S_l2_syminfonce": dict(energy_fn="l2", contrastive_loss_fn="sym_infonce",
                            logsumexp_penalty_coeff=0.1),
}


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="C_l2_infonce", choices=sorted(PRESETS))
    ap.add_argument("--impl", default="warp", choices=["jax", "warp"])
    ap.add_argument("--scene", default="full",
                    choices=["full", "full035", "mesh", "allprim"],
                    help="full = 0.40 m bar spacing (all earlier runs); full035 = same "
                         "asset with 0.35 m spacing (M3.1 easy geometry); mesh/allprim = "
                         "capsule MJX variants (different keyframe, not used for runs)")
    ap.add_argument("--steps", type=int, default=50_000_000)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--num-eval-envs", type=int, default=16)
    ap.add_argument("--episode-length", type=int, default=501)   # 501 x 0.02 s ~ 10 s
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-goal-bar", type=int, default=4,
                    help="fixed instruction goal for evaluation; -1 = random like training")
    ap.add_argument("--train-goal-bar-min", type=int, default=0,
                    help="training instruction goals are sampled uniformly from "
                         "[min, 5); set 1 to exclude B0 (= the bar the robot starts on)")
    ap.add_argument("--train-goal-bar", type=int, default=-1,
                    help="FIXED training instruction goal (e.g. 1 = always 'go to B1'); "
                         "-1 = sample per episode using --train-goal-bar-min. "
                         "Overrides --train-goal-bar-min when >= 0.")
    ap.add_argument("--action-window", default="reach", choices=["m1", "reach"],
                    help="joint-target window per unit action (rad). 'm1' = the +-0.2 rad "
                         "placeholder used by the first runs -- MEASURED to be unable to "
                         "express a reaching pose (best |p_R-B1| 0.29 m vs 0.05 needed); "
                         "'reach' = sized from the M3.0 IK solve. See M3.0b in "
                         "docs/执行计划.md")
    ap.add_argument("--goal-variant", default="full",
                    choices=["full", "support", "support_hold", "support_dual",
                             "position", "cross", "cross3", "hold2", "advance"],
                    help="full = [x,z,p_L(3),p_R(3),c_L,c_R] (10-D, requires BOTH hands); "
                         "support = [...,max(c_L,c_R)] (9-D, releasing ONE hand is free but "
                         "losing the last grip is penalised); support_hold = support + the "
                         "sustained-contact entry 1-exp(-streak/10) (10-D) so that merely "
                         "*touching* the target bar no longer satisfies the goal -- the "
                         "terminal state has to be one the robot can stay in; "
                         "position = [x,z,p_L(3),p_R(3)] (8-D, grasp ignored -> the policy "
                         "can learn to hang without grasping).  See "
                         "docs/M2_JaxGCRL接入记录.md 9.22/9.25/9.34.  Must match at eval time.")
    ap.add_argument("--start-bar-max", type=int, default=0,
                    help="M4: reset on a uniformly random bar in [0, k] by translating "
                         "the robot by k*spacing (bars are periodic, so the physics is "
                         "identical).  Requires --goal-variant advance, whose goal is "
                         "bar-independent.")
    ap.add_argument("--goal-ahead", type=int, default=0,
                    help="M4: sample the goal bar strictly AFTER the bar the episode "
                         "started on (needs a sampled goal bar, e.g. --train-goal-bar -1). "
                         "Combine with --start-bar-max N-2 to start on any of the first "
                         "N-1 bars and always aim forward.")
    ap.add_argument("--goal-ahead-max", type=int, default=0,
                    help="cap on how many bars ahead the sampled goal bar may be "
                         "(0 = no cap).  --goal-ahead-max 1 = the instruction goal is "
                         "always exactly the next bar: uniformly difficult, always "
                         "reachable, while multi-bar progress still has to come from "
                         "the relabelled goals.")
    ap.add_argument("--goal-position-only", type=int, default=0,
                    help="deprecated alias for --goal-variant position")
    ap.add_argument("--discounting", type=float, default=0.995)
    ap.add_argument("--expl-hold", type=int, default=1,
                    help="hold one exploration perturbation for K control steps "
                         "(1 = i.i.d. per step, upstream). 5/10/20 gives the random "
                         "exploration a 0.1-0.4 s time correlation, which a swing needs")
    ap.add_argument("--entropy-param", type=float, default=0.5,
                    help="target entropy = -entropy_param * action_size (upstream 0.5). "
                         "Raise it when the action window grows, so the JOINT-space "
                         "exploration noise stays put: reach vs m1 needs ~2.2")
    ap.add_argument("--min-replay-size", type=int, default=1000)
    ap.add_argument("--max-replay-size", type=int, default=20000,
                    help="replay capacity in PER-ENV time slices; the buffer is "
                         "(max_replay_size, num_envs, 169) so memory is x num_envs")
    ap.add_argument("--buffer-gb", type=float, default=2.0,
                    help="device-memory cap for the replay buffer; --max-replay-size "
                         "is clamped down to fit (JaxGCRL's 100000 default needs "
                         "8.06 GiB at 128 envs and OOMs an 8 GB card). NB: JAX "
                         "preallocates 75%% of the GPU by default -- raise "
                         "XLA_PYTHON_CLIENT_MEM_FRACTION to push this higher")
    ap.add_argument("--unroll-length", type=int, default=62)
    ap.add_argument("--n-frames", type=int, default=10)          # 50 Hz control
    ap.add_argument("--naconmax-per-world", type=int, default=128,
                    help="MJX-Warp contact budget PER WORLD; the env gets "
                         "per_world*num_envs because Warp's naconmax is a global "
                         "budget and a too-small value silently drops contacts")
    ap.add_argument("--njmax", type=int, default=512, help="constraint rows per world")
    ap.add_argument("--xla-mem-fraction", type=float, default=0.70,
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION.  JAX preallocates this share "
                         "of the GPU up front, and MJX-Warp allocates ITS buffers from "
                         "outside that pool -- on an 8 GB card the default 0.75 left "
                         "Warp too little room and the eval step died with Warp's own "
                         "out-of-memory.  0 disables the override (JAX default 0.75)")
    ap.add_argument("--ls-iterations", type=int, default=0,
                    help="0 = keep the scene XML value (20); 50 = MuJoCo's default "
                         "(the solver line-search cap is hit at 20 -> LS_ITERATIONS overflow)")
    ap.add_argument("--n-hidden", type=int, default=17, help="trunk Dense layers. 17 = the paper's "
                    "depth 16 (its stem is a separate layer that JaxGCRL folds into this count), "
                    "i.e. stem + 4 residual blocks x 4 Dense")          # network depth D
    ap.add_argument("--h-dim", type=int, default=256)            # network width W
    ap.add_argument("--repr-dim", type=int, default=64)
    ap.add_argument("--skip-connections", type=int, default=4)   # ResBlock = 4 Dense
    ap.add_argument("--use-ln", type=int, default=1,
                    help="LayerNorm in the two critic encoders (upstream field; note the "
                         "Actor class has the field but upstream's CRL never sets it)")
    ap.add_argument("--num-evals", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--wandb-project", default="monkeybars-crl")
    ap.add_argument("--wandb-group", default=".")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--checkpoint-dir", default=None, help="where to save final params")
    ap.add_argument("--save-every", type=int, default=10,
                    help="with --checkpoint-dir: save an actor-only checkpoint every N "
                         "EVALS (0 = only at the end).  A run whose params are not "
                         "saved can never be turned into a video afterwards.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 4k env steps, 4 envs, no eval spam")
    return ap.parse_args()


def git_rev() -> str:
    """Short commit hash of the tree this run is executed from (``-dirty`` if modified).

    Recorded in ``args.json`` and in every checkpoint so a result can always be
    traced back to the exact source version (see docs/M2_JaxGCRL接入记录.md 9.34.4).
    Never raises: falls back to "unknown" outside a git checkout.
    """
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=5)
        if head.returncode != 0:
            return "unknown"
        rev = head.stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                            capture_output=True, text=True, timeout=5)
        if st.returncode == 0 and st.stdout.strip():
            rev += "-dirty"
        return rev or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    args = parse()
    args.git_commit = git_rev()
    if args.smoke:
        args.steps = 4000
        args.num_envs = 4
        args.num_eval_envs = 4
        args.episode_length = 101
        args.batch_size = 100
        args.min_replay_size = 50
        args.unroll_length = 20
        args.num_evals = 3
    exp = args.exp_name or f"brach_{args.preset}_{args.impl}"
    run_dir = os.path.join(REPO, "runs", exp)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    # JAX reads this when it first initialises the GPU allocator, so it must be set
    # BEFORE any `import jax` (nothing above imports jax: mjx_backend only needs
    # numpy, and enable_warp_compat only imports warp).
    if args.xla_mem_fraction > 0:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)

    if args.impl == "warp":
        mb.enable_warp_compat()

    # JaxGCRL's CRL calls ``brax.envs.training.wrap``; swap in the MJX-safe one
    # (AutoResetWrapper must tolerate empty / non-batched mjx.Data leaves).
    import brax.envs as brax_envs
    from envs.brax_ext import wrap as mjx_wrap
    brax_envs.training.wrap = mjx_wrap

    from envs.brachiation import create_brachiation
    from jaxgcrl.agents.crl import CRL
    from jaxgcrl.utils.config import Config, RunConfig

    goal_variant = "position" if args.goal_position_only else args.goal_variant
    if args.goal_position_only and args.goal_variant != "full":
        print(f"[warn] --goal-position-only overrides --goal-variant {args.goal_variant}")
    env_kwargs = dict(impl=args.impl, scene=args.scene, n_frames=args.n_frames,
                      goal_bar_min=args.train_goal_bar_min,
                      goal_variant=goal_variant,
                      action_window=args.action_window,
                      start_bar_max=args.start_bar_max,
                      goal_ahead=bool(args.goal_ahead),
                      goal_ahead_max=int(args.goal_ahead_max),
                      njmax=args.njmax,
                      naconmax=max(1024, args.naconmax_per_world * args.num_envs))
    # variants with ONE bar-independent instruction goal (no --train-goal-bar)
    is_cross = goal_variant in ("cross", "cross3", "hold2", "advance")
    if is_cross:
        print(f"[goal] variant={goal_variant} has ONE fixed instruction goal; "
              "--train-goal-bar/--eval-goal-bar are ignored")
        env_kwargs["goal_bar"] = None          # ignored by reset() for this family
    elif args.train_goal_bar >= 0:
        # fixed instruction goal for training (e.g. M3: always "go to B1")
        env_kwargs["goal_bar"] = int(args.train_goal_bar)
    if args.ls_iterations:
        env_kwargs["ls_iterations"] = args.ls_iterations
    train_env = create_brachiation(**env_kwargs)
    # The evaluator resets ``eval_env`` itself, so a random per-episode goal there
    # would make ``eval/episode_success`` incomparable across runs.  Pin the
    # evaluation instruction goal to a fixed bar (default B4 = the far bar).
    eval_kwargs = dict(env_kwargs)
    if not is_cross:
        eval_kwargs["goal_bar"] = None if args.eval_goal_bar < 0 else args.eval_goal_bar
    # The evaluation must stay a FIXED probe so `eval/*` is comparable across evals
    # and across runs: always start on bar 0, and drop the forward-only goal sampling
    # when the goal bar is pinned (goal_ahead requires a sampled goal bar).  With
    # `--eval-goal-bar 4` this makes the probe "from B0, traverse to B4", i.e. a clean
    # multi-bar readout (`advance_max` = bars actually advanced in that probe), while
    # the *training* distribution keeps the randomised start + forward goal.
    eval_kwargs["start_bar_max"] = 0
    if eval_kwargs.get("goal_bar") is not None:
        eval_kwargs["goal_ahead"] = False
    # Warp's naconmax is a GLOBAL budget, so the eval env must scale it by ITS OWN
    # world count -- sharing the training value (128 x 128 = 16384) with only 16
    # eval worlds wasted 8x the contact memory and was what ran the 8 GB card out
    # of Warp memory during evaluation.
    eval_kwargs["naconmax"] = max(1024, args.naconmax_per_world * args.num_eval_envs)
    eval_env = create_brachiation(**eval_kwargs)
    print(f"[explore] expl_hold={args.expl_hold} control steps "
          f"({args.expl_hold * args.n_frames * 0.002:.2f} s per perturbation)")
    print(f"[entropy] target = -{args.entropy_param} x {train_env.action_size} "
          f"= {-args.entropy_param * train_env.action_size:.2f}")
    print(f"[action] window={train_env.action_window} "
          f"scales={np.round(np.asarray(train_env.act_scale), 3).tolist()} (rad per unit action)")
    print(f"[goal] variant={train_env.goal_variant} goal_size={train_env.goal_size} "
          f"state_dim={train_env.state_dim} obs={train_env.observation_size} "
          f"indices={train_env.goal_indices}")
    if is_cross:
        instr = f"FIXED single goal ({train_env.goal_variant})"
        ev_instr = "same single goal"
    else:
        instr = (f"FIXED B{env_kwargs.get('goal_bar')}" if args.train_goal_bar >= 0
                 else f"U[{args.train_goal_bar_min}, {train_env.n_bars})")
        ev_instr = f"B{eval_kwargs.get('goal_bar')}"
    print(f"[goal] train instruction = {instr} | eval instruction = {ev_instr} "
          f"(envs={args.num_eval_envs}, naconmax={eval_kwargs['naconmax']})")
    print(f"[env] {train_env.backend} scene={args.scene} action={train_env.action_size} "
          f"obs={train_env.observation_size} state_dim={train_env.state_dim} "
          f"n_frames={args.n_frames} ({1/(args.n_frames*train_env.mj_model.opt.timestep):.0f} Hz) "
          f"naconmax={env_kwargs['naconmax']} njmax={env_kwargs['njmax']} "
          f"ls_iterations={train_env.ls_iterations}")
    print(f"[mem] XLA_PYTHON_CLIENT_MEM_FRACTION="
          f"{os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '(jax default 0.75)')} "
          f"(MJX-Warp allocates outside this pool -- leave it room on an 8 GB card)")

    cfg = PRESETS[args.preset]

    # ---- replay-buffer memory guard ------------------------------------- #
    # JaxGCRL's TrajectoryUniformSamplingQueue stores
    #     data : (max_replay_size, num_envs, data_size)
    # so `max_replay_size` is a PER-ENV capacity and the total is x num_envs.
    # Measure the flattened transition exactly the way the buffer does.
    import jax.numpy as _jnp
    from jax import flatten_util as _fu

    _dummy = {
        "observation": _jnp.zeros((train_env.observation_size,)),
        "action": _jnp.zeros((train_env.action_size,)),
        "reward": 0.0, "discount": 0.0,
        "extras": {"state_extras": {"truncation": 0.0, "traj_id": 0.0}},
    }
    data_size = int(len(_fu.ravel_pytree(_dummy)[0]))
    per_slice = data_size * 4 * args.num_envs          # bytes per time slice
    cap = max(1, int(args.buffer_gb * 2 ** 30 / per_slice))
    if args.max_replay_size > cap:
        print(f"[buffer] --max-replay-size {args.max_replay_size} x {args.num_envs} envs "
              f"= {args.max_replay_size * per_slice / 2 ** 30:.2f} GiB > "
              f"--buffer-gb {args.buffer_gb}: clamping to {cap}")
        args.max_replay_size = cap
    args.min_replay_size = min(args.min_replay_size, args.max_replay_size)
    print(f"[buffer] max_replay_size={args.max_replay_size} (per env) x {args.num_envs} "
          f"envs x {data_size} floats = "
          f"{args.max_replay_size * args.num_envs / 1e6:.2f}M transitions, "
          f"~{args.max_replay_size * per_slice / 2 ** 30:.2f} GiB "
          f"| min_replay_size={args.min_replay_size}")
    # record the *effective* values (args.json was written before the guard)
    with open(os.path.join(run_dir, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    agent = CRL(
        policy_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
        batch_size=args.batch_size,
        discounting=args.discounting,
        logsumexp_penalty_coeff=cfg["logsumexp_penalty_coeff"],
        max_replay_size=args.max_replay_size,
        min_replay_size=args.min_replay_size,
        unroll_length=args.unroll_length,
        h_dim=args.h_dim, n_hidden=args.n_hidden,
        skip_connections=args.skip_connections,
        use_ln=bool(args.use_ln), use_relu=USE_RELU,
        repr_dim=args.repr_dim,
        contrastive_loss_fn=cfg["contrastive_loss_fn"],
        energy_fn=cfg["energy_fn"],
        entropy_param=args.entropy_param,
        expl_hold=args.expl_hold,
    )
    run = RunConfig(
        env="brachiation", total_env_steps=args.steps,
        episode_length=args.episode_length,
        num_envs=args.num_envs, num_eval_envs=args.num_eval_envs,
        num_evals=args.num_evals, action_repeat=1,
        seed=args.seed, backend=None, log_wandb=False, cuda=(args.impl == "warp"),
    )
    agent.check_config(run)
    print(f"[net] W={args.h_dim} D={args.n_hidden} skip={args.skip_connections} "
          f"repr_dim={args.repr_dim} activation={'relu' if USE_RELU else 'swish'} "
          f"LN(encoders)={bool(args.use_ln)} (Actor has no LN switch upstream)")
    print(f"[loss] preset={args.preset} {cfg}")
    print(f"[run] git={args.git_commit} steps={run.total_env_steps} num_envs={run.num_envs} "
          f"episode_length={run.episode_length} batch={args.batch_size} gamma={args.discounting}")

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, group=args.wandb_group, name=exp,
                   config={**vars(args), **cfg}, mode=args.wandb_mode,
                   dir=run_dir)
        print(f"[wandb] project={args.wandb_project} name={exp} mode={args.wandb_mode}")

    rows = []
    nan_evals = 0          # consecutive evals with NaN training metrics (guard below)
    csv = os.path.join(run_dir, "progress.csv")

    def save_actor(dest, actor_params, steps):
        """Actor-only checkpoint: params + everything needed to rebuild them.

        ``src/render_policy.py`` reads this dict; renderability is why we save it
        (a finished run whose params were never written cannot be recorded).
        """
        import pickle
        os.makedirs(dest, exist_ok=True)
        blob = {
            "actor": actor_params,
            "steps": int(steps),
            "config": dict(h_dim=args.h_dim, n_hidden=args.n_hidden,
                           skip_connections=args.skip_connections, repr_dim=args.repr_dim,
                           action_size=train_env.action_size, n_frames=args.n_frames,
                           scene=args.scene, impl=args.impl, n_bars=train_env.n_bars,
                           state_dim=train_env.state_dim,
                           goal_variant=str(train_env.goal_variant),
                           goal_size=int(train_env.goal_size),
                           action_window=str(train_env.action_window),
                           git_commit=str(args.git_commit)),

        }
        tag = f"actor_{int(steps):09d}.pkl" if steps >= 0 else "actor_final.pkl"
        for name in (tag, "actor_latest.pkl"):
            with open(os.path.join(dest, name), "wb") as fh:
                pickle.dump(blob, fh)
        print(f"[ckpt] {os.path.join(dest, tag)}", flush=True)

    def progress(num_steps, metrics, make_policy=None, params=None, env=None,
                 do_render=False):
        """Signature must match jaxgcrl/utils/env.py::MetricsRecorder.progress.

        JaxGCRL passes ``(num_steps, metrics, make_policy, params, unwrapped_env,
        do_render=...)`` where ``params`` are the ACTOR params of this eval step.
        We use that to write renderable checkpoints (``--save-every``), because a
        run that finishes without one cannot be turned into a video afterwards.
        """
        rows.append({"steps": int(num_steps), **{k: float(v) for k, v in metrics.items()}})
        keys = [k for k in ("eval/episode_success", "eval/episode_dist",
                            "eval/episode_max_bar", "training/critic_loss", "training/sps")
                if k in metrics]
        print(f"  [{num_steps:>9}] " + "  ".join(f"{k.split('/')[-1]}={float(metrics[k]):.4g}"
                                                 for k in keys), flush=True)
        if args.wandb:
            import wandb
            wandb.log({k: float(v) for k, v in metrics.items()}, step=int(num_steps))
        with open(csv, "w") as fh:
            allk = sorted({k for r in rows for k in r})
            fh.write(",".join(allk) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(k, "")) for k in allk) + "\n")
        if args.checkpoint_dir and params is not None and args.save_every > 0:
            n = len(rows) - 1
            if n % args.save_every == 0:
                save_actor(args.checkpoint_dir, params, int(num_steps))

        # NaN guard: a diverged critic (logits = -inf -> inf - inf) poisons the whole
        # run, and on the GPU that only shows up 1.5 h later.  Treat 3 consecutive
        # NaN evals as a dead run and stop, keeping the checkpoints written so far.
        nonlocal nan_evals
        bad = any(metrics.get(k) != metrics.get(k) for k in
                  ("training/critic_loss", "training/actor_loss", "training/logits_pos"))
        nan_evals = nan_evals + 1 if bad else 0
        if bad:
            print(f"  [warn] NaN training metrics ({nan_evals}/3 consecutive) -- "
                  f"see docs/M2_JaxGCRL接入记录.md 9.34.5", flush=True)
        if nan_evals >= 3:
            print("[abort] training diverged to NaN; checkpoints kept.  Stopping.", flush=True)
            raise SystemExit(2)

    t0 = time.time()
    train_fn, params, _ = agent.train_fn(train_env=train_env, eval_env=eval_env,
                                         config=run, progress_fn=progress)
    if args.wandb:
        import wandb
        wandb.finish()
    if args.checkpoint_dir and params is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        from brax.io import model as brax_model
        brax_model.save_params(os.path.join(args.checkpoint_dir, "final"), params)
        # also an actor-only dict (same format as --save-every) so
        # src/render_policy.py can render either artifact
        actor_params = params[1] if isinstance(params, (tuple, list)) else params
        save_actor(args.checkpoint_dir, actor_params,
                   int(rows[-1]["steps"]) if rows else -1)
        print(f"[ckpt] {args.checkpoint_dir}/final (+ actor_*.pkl)")
    print(f"[done] {time.time()-t0:.1f}s wall; progress -> {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
