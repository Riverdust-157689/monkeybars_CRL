# monkeyBars_CRL — 宇树 G1 猴杠（brachiation）× 对比强化学习（CRL）

用 **MuJoCo MJX（Warp 后端）** 直接做物理、用 **JaxGCRL 的 CRL**（无奖励、目标条件、
future-state 重标记 + InfoNCE）训练 Unitree G1 在 5 根平行杆之间"荡杆前进"。

* 机器人：`g1_with_hands.xml`（Menagerie），43 个驱动关节 + 6 自由度浮动基座，`nq=50`，34.4 kg。
* 场景：`assets/g1_brachiation/scene_bars*.xml`（程序化生成，`build_scene.py`）。
* 物理：`mujoco.mjx` + `impl="warp"`（全 mesh）；brax 只当类型库用。
* 训练：`src/train.py` + `third_party/jaxgcrl`（上游 + `patches/` 里的补丁）。

## 快速开始（新机器）

```bash
git clone <repo-url> monkeyBars_CRL && cd monkeyBars_CRL
tools/setup_env.sh                    # venv + 依赖 + jaxgcrl + 网格 + 资产校验 + 静态检查
.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant support_dual --train-goal-bar 1 --eval-goal-bar 1 \
  --expl-hold 10 --num-evals 20 --steps 12200000 \
  --checkpoint-dir runs/ckpt --save-every 5 \
  --impl warp --scene full035 --wandb --exp-name demo
```

详细步骤、GPU/Warp 注意事项、CPU 回放、以及"仓库里没有的东西怎么拿"见
👉 **`docs/复现环境.md`**。

## 目录

| 路径 | 内容 |
|---|---|
| `src/envs/brachiation.py` | 环境本体：MjxModel/Data、goal 变体、动作窗、指标与覆盖率诊断 |
| `src/train.py` | 训练入口（JaxGCRL CRL + brax 包装 + wandb + checkpoint） |
| `src/render_policy.py` | 载入 checkpoint 回放 → GIF/mp4 + `trace.csv`（本机 CPU 亦可） |
| `src/check_args.py` / `check_metrics.py` | 静态检查（跑 GPU 前必跑） |
| `src/check_reset.py` / `check_scene.py` / `smoke_env.py` | reset 容差、资产校验、环境冒烟 |
| `src/goal_geometry.py` / `m3_swing.py` | goal 空间几何、摆动/IK 可达性（纯 MuJoCo，CPU） |
| `assets/g1_brachiation/` | 场景生成脚本 + 导出的场景 XML（网格另取） |
| `third_party/jaxgcrl/` | **vendored 的上游 JaxGCRL 源码**（只含 Python 包 ~400 KB + LICENSE + PROVENANCE.md）|
| `patches/` | 对上游 JaxGCRL 的补丁（出处的单一真相，见 `third_party/jaxgcrl/PROVENANCE.md`） |
| `tools/` | `setup_env.sh`（复现环境）、`make_patch.sh`（重新生成/校验补丁） |
| `docs/` | 执行计划、接入记录（全部实验日志）、任务变量定义、M4 goal 讨论 |

## 不随仓库分发的东西

| 内容 | 为什么 | 怎么拿 |
|---|---|---|
| `.venv*` | 1–2 GB | `tools/setup_env.sh` |
| `third_party/jaxgcrl` 的**其余部分** | 上游的 `.git`、36 MB benchmark 网格、docs/notebooks/tests | 不需要 |
| `runs/` | 训练产物（checkpoint/progress.csv/渲染） | 自行 `rsync`；每个 `runs/<exp>/args.json` 里记了 `git_commit` 与全部超参 |
| `assets/g1_brachiation/menagerie/` | 上游网格（38 MB） | `tools/setup_env.sh` + `menagerie_sha256.txt` 校验 |
| `ref papers/` | 参考文献 PDF | 不需要 |

## 为什么 jaxgcrl 是 vendored，而不是 fork / submodule

我们只用到上游的 `jaxgcrl.agents.crl` 与 `jaxgcrl.utils`（物理完全是我们自己的），
而它的 Python 源码只有 ~400 KB（那 78 MB 里 36 MB 是上游自己 benchmark 用的 Franka 网格）。
所以选择：**把源码 vendor 进主仓库**（`third_party/jaxgcrl/`，随 `git clone` 一起来，
新机器不需要联网、不需要打补丁），同时用 `patches/jaxgcrl_crl_losses.patch`（128 行、
只碰 2 个文件）作为**差异与出处的单一真相**。

* **不 fork**：我们并不打算给上游提 PR；fork 只会多一个要维护的仓库，且新机器还要联网 clone fork。
* **不用 submodule**：submodule 记的是上游 commit，我们这两处改动还是得靠补丁，等于 submodule + patch 两套机制叠加。
* **要重新同步上游**时按 `third_party/jaxgcrl/PROVENANCE.md` 的步骤（clone 上游 @5a6e7a0 → `git apply` 补丁 → `diff -r` 校验 vendored 树）。

## 版本管理约定

* 每个实验一个 git 分支（`main` = 基线，`m3/*`、`m4/*` = 各轮实验）。
* 每次训练会把 **`git_commit`（含 `-dirty`）写进 `args.json` 和 checkpoint 的 config**，
  渲染器也会打印它 ⇒ 任何结果都能回溯到确切的代码版本。
* 改过 `third_party/jaxgcrl` 之后必须跑 `tools/make_patch.sh`（会 `git apply -R --check` 自检）。
* 单 seed 的结论要谨慎（所有 run 都是 `--seed 0`）。
