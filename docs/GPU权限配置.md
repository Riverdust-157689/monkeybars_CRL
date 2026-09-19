# 让沙箱内可见 GPU（给 DSH 自己跑训练/渲染用）

## 现状：不是权限问题，是**设备节点不存在**

DSH 的本地沙箱固定用 bwrap 这样建：

```js
// @deepseek-ai/dsh-sandbox-local/lib/index.js::bwrapProfileArgs()
"--ro-bind", "/", "/",
"--dev", "/dev",            // ← 关键：新建一个"最小 /dev"
"--unshare-pid", "--proc", "/proc", "--die-with-parent",
// workspace-write 时再加 --tmpfs /tmp 与 --bind <workspace> <workspace>
```

`--dev /dev` 会造一个**全新的最小 devtmpfs**，里面只有 `null/zero/random/tty/...`。
所以在沙箱里：

```
$ ls /dev            → core fd full null ptmx pts random shm stderr stdin stdout tty urandom zero
$ ls /dev/nvidia*    → No such file or directory
$ python -c "import jax; print(jax.devices())"  → CUDA_ERROR_NO_DEVICE
```

CUDA 驱动要 `cuInit(0)`，它需要 `/dev/nvidiactl`、`/dev/nvidia0`、`/dev/nvidia-uvm`（以及 EGL 渲染要 `/dev/dri/*`）；这些节点在沙箱里不存在，所以**任何** GPU 程序都起不来。

> ⚠️ 界面上那个 `workspace-write → danger-full-access` 的审批**不会**解决问题：
> 它只是**文件**策略（是否 `--bind` 工作区、`/tmp` 是否可写），
> `bwrapProfileArgs()` 里 `--dev /dev` 是**无条件**加的。
> 批准它只放宽文件读写范围，`/dev` 依旧是最小集合。

## 两种改法

### A（推荐，配置层，能扛住 npx 重装）

思路：让沙箱**共享宿主机的 `/dev`**（`--dev-bind /dev /dev` 而不是 `--dev /dev`）。
provider 支持 `runnerCommand`：一旦设置，它会跳过内置探测，并把
`bwrap` 风格的 profile 参数**原样**传给这个自定义 runner：

```
<runnerCommand...> --ro-bind / / --dev /dev --unshare-pid --proc /proc --die-with-parent [--tmpfs /tmp --bind WS WS] -- <真正的命令>
```

所以只要给一个**包一层 bwrap 的脚本**，把 `--dev /dev` 改写成 `--dev-bind /dev /dev` 即可。

1. 仓库里已经写好这个脚本：`tools/bwrap-dev`（可用 `BWRAP_DEV_DRY=1` 干跑看改写结果）。
   复制到沙箱外，例如：

   ```bash
   mkdir -p ~/bin && cp ~/monkeyBars_CRL/tools/bwrap-dev ~/bin/bwrap-dev && chmod +x ~/bin/bwrap-dev
   # 自检（应打印 --dev-bind /dev /dev）
   BWRAP_DEV_DRY=1 ~/bin/bwrap-dev --ro-bind / / --dev /dev -- true
   ```

2. 在 profile 的 patch 层加上 id 覆盖（`~/.dsh/profiles/web/cordis.patch.yml`，
   原本是 `[]`，把下面这段放进数组里）：

   ```yaml
   - id: sandbox
     config:
       runnerCommand:
         - /home/firedust/bin/bwrap-dev
       runnerFailureSignatures:
         - 'bwrap: '
   ```

3. **开一个新会话**（provider 会把 runner 选择结果缓存到实例上；新会话会重新实例化），
   如果不行就重启 dsh。然后在会话里让我执行：

   ```
   ls -la /dev/nvidia*
   nvidia-smi
   .venv-warp/bin/python -c "import jax; print(jax.devices())"
   ```

   看到 `CudaDevice(id=0)` 就说明 GPU 已经可用。

### B（快速但有副作用：直接改 npx 缓存里的 provider）

```bash
F=/home/firedust/.npm/_npx/1e7f6d9597241db0/node_modules/@deepseek-ai/dsh-sandbox-local/lib/index.js
python3 - "$F" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = '\t\t"--dev",\n\t\t"/dev",'
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, '\t\t"--dev-bind",\n\t\t"/dev",\n\t\t"/dev",'))
print("patched", p)
PY
```
然后**重启 dsh**（该文件在进程启动时就被 import 了）。
缺点：npx 缓存被清掉/换版本时会丢；且这是改依赖包本体，不推荐长期用。

## 放开之后我能做什么（也是为什么要放开）

* 把 4h 训练当**后台任务**跑（`run_in_background`），我边跑边读 `runs/*/progress.csv` 判断趋势；
* 用 `src/render_policy.py` 渲染后，我能**直接看帧**（`read_image`），
  不必再靠你描述"松手了但没有前伸"——我可以自己从画面上量和判断；
* 迭代闭环从"我改代码 → 你跑 → 你描述现象"变成"我改 → 我跑 → 我看 → 我改"。

## 代价与注意

* 沙箱内可以看到宿主机**全部**设备节点（`/dev` 共享）。其余限制不变：
  `/` 只读、写范围仍限工作区（+`/tmp`）、`--unshare-pid` 仍在。
* 我们同时用 GPU 会抢那 8 GB 显存——跑长任务前打个招呼，或者让我在你不用时跑。
* `MUJOCO_GL`：渲染建议 `MUJOCO_GL=egl`（无窗口）；共享 `/dev` 后 `/dev/dri` 也会可见。
* 如果 A 方案下 bwrap 报权限/挂载错误，先 `BWRAP_DEV_DRY=1` 确认改写没跑偏，再看 stderr 是否含 `bwrap: `（这是 `runnerFailureSignatures` 用来判"沙箱坏掉"而不是"命令被拒绝"的）。
