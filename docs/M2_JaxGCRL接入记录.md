# Monkey Bars × Contrastive RL（Unitree G1）—— 任务评估与执行计划

> ⚠️ **文档状态（请先读这一段）**
> 本文件 §1–§8 与附录 A/B 是**最初版执行计划的副本，已落后**（例如 §1.2 的"杆高 z≈2.0 m"、§1.4 的"k=50 步"、§6 的 M2 清单都已过时）。
> - 计划的**权威最新版**：`docs/执行计划.md`
> - 任务变量（动作/状态/goal/终止/reset/指标）的**权威最新版**：`docs/M2_任务变量定义.md`（v3）
> - 本文件真正有用的增量内容全部在 **第 9 节（§9.1 起，文末）**：接入 JaxGCRL 的实测记录、踩过的坑与修法、损失函数配置、4060 上的运行情况。
> 下面 §1–§8 保留仅作历史参考。

> 输入依据：`docs/思路.md`（研究命题）、`docs/末端执行器设计.md`（抓握抽象）、`docs/仿真与软件平台选择.md`（软件栈）。
> 本版修正：**机器人不再简化**，直接用完整的 Unitree G1（MuJoCo Menagerie 资产，29-DoF 本体 + Dex3-1 三指手 = 43 个主动关节）。因此本文的重点，按你的要求，放在**关节控制分配**上：哪些关节进动作空间、哪些锁死、哪些保持定值、哪些做成被动/欠驱动。
> 本文档中所有 G1 数值均来自实际解析的资产文件（见附录 A），非论文转述。

---

## 0. 结论摘要（TL;DR）

| 维度 | 评估 | 依据 |
|---|---|---|
| 研究命题本身 | **成立且有位置**。现有 brachiation 工作都给了 waypoint / privileged expert / phase schedule / passive hook；"只给一个最终目标、无 reward、无 demo、无 waypoint、无 phase、可主动松手"这一组合确实是空白 | `思路.md` §"与现有工作相比"，两篇 2026 工作 |
| 任务物理可行性 | **可行，但约束很强**：G1 只能以"近直臂悬垂 + 摆动"的方式挂住，不能静态单手支撑偏离垂直线的身体（肩/肘各 25 N·m，单手悬垂允许的质心水平偏移仅 ≈8 cm） | 本文 §2.2 力矩预算 |
| 最大物理风险 | **原装 Dex3-1 手指力矩不够**（1.4–2.45 N·m/关节），双手悬垂时单指静态力矩需求已 ≈2.1 N·m，单手悬垂 ≈4.2 N·m。必须靠"钩挂式形封闭"、或提高仿真力矩上限（记为偏差）、或换定制 1-DoF 夹爪 | §2.2 / §3.4 / R1 |
| 关节控制分配 | 可以给出干净的答案：v1 = **14 维动作**（10 臂 + 2 腰 + 2 抓握）；腿保持定值伺服、腕 pitch/yaw 用等式锁定、手 7 DoF → 1 DoF synergy | §3 全节 |
| 算法风险 | **中高，且这是研究本身**。14 维动作 + 接触切换 + 无 reward，即使对 CRL 也很激进。必须先接受"可能只到 B1"作为有效结论 | R4；`思路.md` 自己就是这么定位的 |
| 软件栈 | 维持 `docs/仿真与软件平台选择.md` 的结论：**JaxGCRL + MJX-JAX** 为主；但本项目要多做两件事：① 显式接触对 + 图元碰撞代理（G1 的 MJX 变体只覆盖了脚-地）；② 设一个吞吐门限，不达标就切 MJX-Warp / MuJoCo Playground | §4 |
| 预期首个可信结果 | **M1（物理保真与握力可行性）约 1–2 周**就能判定这条路能不能走：抓到杆不代表抓得住 | §6 |
| 最坏情况的产出 | "CRL 在接触切换型动力学上的探索边界"本身是可发表的负面结果，而且比自行车干净 | `思路.md` 收尾段 |

**一句话结论**：任务值得做，但**成败不取决于 CRL，取决于末端执行器的力/几何闭合能力**。因此执行计划把"握力与力矩预算的实测验证"提到 CRL 之前，作为第一个硬门限。

---

## 1. 任务定义与判定口径

### 1.1 研究命题

$$
\boxed{\text{Can contrastive goal-conditioned RL autonomously discover contact-switching dynamic locomotion?}}
$$

具体到本任务：只给一个最终目标状态（"稳定悬挂在最后一根杆"），不给前进 reward、不给抓握 reward、不给 waypoint、不给 demonstration、不给 phase scheduled，让 CRL 自己长出

$$
\mathcal M_{LR}\to\mathcal M_L\to\mathcal M_{LR'}\to\mathcal M_{R'}\to\cdots
$$

的接触模式切换链。

### 1.2 场景规格（v1）

| 项 | 取值 | 说明 |
|---|---|---|
| 杆数 | 5（B0…B4） | 第一版；可退化为 2 根做 M3 |
| 杆间距 $d$ | 0.40 m（易化版 0.30 m） | 与臂长比较：肩→腕 0.354 m + 手 ≈0.1 m，因此必须靠摆动才能到下一根 —— 这正是任务 |
| 杆几何 | 半径 0.025 m 的 **capsule**，沿 Y 轴水平放置，刚性固定在世界 | 用 capsule 不用 cylinder：MJX-JAX 下 cylinder 只能与"图元"碰撞，capsule 全支持（见 §4.2） |
| 杆高度 | 世界坐标 z ≈ 2.0 m | 无地面 |
| 地面 | **不建地面**（避免接触对与无关碰撞） | 掉落判据靠根节点高度 |
| 初态 $s_0$ | 双手抓 B0，$\dot q_0=0$ | 需要 IK 解出一个双臂同时到杆的对称悬挂构型（M0 任务） |
| 终态 $g^*$ | 稳定悬挂于 B4 | goal 的精确形式见 §1.4 |
| 终止 | 根节点 z < 杆高 − 1.0 m，或 episode 长度上限 | 掉落终止；不回退到 reward |

### 1.3 与现有工作的差异（保持 `思路.md` 的定位）

| | Iwata et al. 2026 | ETH humanoid sparse traversal | 本项目 |
|---|---|---|---|
| 末端 | — | passive hook | 主动可松开的抓握 |
| 指导 | waypoint-guided | privileged expert + phase scheduler + teacher-student | 无 |
| reward | success + energy | 任务 reward | 无（只有 goal） |
| 本体的简化 | 专用双臂 | 完整 humanoid | **完整 G1，不做形态简化**（本项目决策） |

代价与收益要写清楚：保留完整 G1 意味着 43 个主动关节、32.2 kg 的双腿死摆、真实 mesh 资产与真实力矩上限；换来的是无需自建模型、结果可对真机、以及"欠驱动来自浮动基座+接触切换"这一干净叙事。

### 1.4 goal 的定义（这是最容易做错的地方）

`思路.md` 倾向 `g = [p_torso, p_L, p_R]`。本计划的建议：

$$
g=[x_{\text{root}},\ z_{\text{root}},\ \text{onehot}(\text{左手所抓杆}),\ \text{onehot}(\text{右手所抓杆}),\ v_{\text{root}}]
$$

理由：
1. **接触状态必须进 goal**。终态定义是"挂着"，不是"手在杆附近"；否则"伸手到 B4 附近然后掉落"会成为最优解。这不是 reward shaping，只是把目标状态说完整。
2. **不要进 goal 的量**：关节角全量 $q$（否则过约束，几乎不可能精确匹配）、以及任何与"过程"相关的量（摆动幅度、单手时刻）。goal 里出现过程量等于告诉它技能分解。
3. 目标不是一个点，而是**一个集合**：$g^*$ 取"B4 标准悬挂姿态 ± 小扰动"，避免对单帧过拟合。
4. **成功判据与 goal 分离**：
   - `success`：连续 $k=50$ 步（≈0.2 s）满足"至少一只手抓在 B_N"且根节点速度 < ε —— 即"稳定悬挂"，避免"扑向目标后立刻掉落"这种 CRL 经典退化解（自行车项目里已经踩过这个坑）。

### 1.5 评价指标（也是论文的图）

| 指标 | 用途 |
|---|---|
| $P(\max\ \text{bar index}\ge k)$ vs env steps，$k=1..4$ | **头条图**：自生成课程的可视化 |
| 首次单手释放时间 / 首次抓到自己以外杆的时间 | 技能涌现时间线 |
| 双手→单手→双手 的接触模式切换次数、掉落率 | hybrid dynamics 的量化 |
| B_k 到达时间、每 episode 最大前进距离 | 效率 |
| 摆动幅度与相位、泵能（腰部/肩部做功）时间曲线 | 机制分析，支撑"它是怎么做到的" |
| 目标占据率 / 逗留成功率 | 与成功率并列报告，防止退化解 |

---

## 2. G1 平台评估（本计划的核心之一）

### 2.1 资产与规格（实测）

MuJoCo Menagerie `unitree_g1`（BSD-3-Clause）提供三种资产，本项目要用的是"**带手 + MJX 友好**"的组合，而这个组合上游并没有现成的：

| 资产 | 内容 | 本项目可用性 |
|---|---|---|
| `g1.xml` | 29-DoF，无手，位置伺服 kp=500（未限力） | 参考 |
| `g1_with_hands.xml` | **43 关节**：腿 6×2 + 腰 3 + 臂 7×2 + 手 7×2，含 `actuatorfrcrange` | **形态基座** |
| `g1_mjx.xml` + `scene_mjx.xml` | 29-DoF，**capsule 碰撞几何 + 显式 `<pair>` 列表 + 调过的 solver/PD**，已上真机（MuJoCo Playground） | **MJX 改造模板** |

**43 个关节清单**（`docs` 与代码请以此命名为准）：

```text
float base(6, 无 actuator)
腿 12:  {left,right}_hip_{pitch,roll,yaw} / knee / ankle_{pitch,roll}
腰  3:  waist_{yaw,roll,pitch}
臂 14:  {left,right}_shoulder_{pitch,roll,yaw} / elbow / wrist_{roll,pitch,yaw}
手 14:  {left,right}_hand_{thumb_0,thumb_1,thumb_2,middle_0,middle_1,index_0,index_1}
```

**质量与几何**（解析 Menagerie MJCF 所得）：

| 量 | 值 |
|---|---|
| 全机质量 | **32.24 kg** |
| 躯干 + 双臂 + 双手 | 14.35 kg |
| 骨盆 + 双腿 | ≈17.9 kg（pelvis 2.86，单腿 7.51） |
| 肩 → 肘 / 肘 → 腕 | 0.184 m / 0.170 m |
| 髋 → 膝 / 膝 → 踝 | 0.311 m / 0.317 m |
| 肩相对骨盆高度 | 0.279 m |
| 腰→腿质心（0 位形） | ≈0.36 m |

**关节力矩上限与行程**（`actuatorfrcrange` / mjlab 电机参数）：

| 关节 | 力矩上限 (N·m) | 行程 (rad) |
|---|---|---|
| hip_pitch / hip_yaw | ±88 | −2.53 … 2.88（pitch） |
| hip_roll / knee | ±139 | — |
| ankle_pitch / ankle_roll | ±50 | −0.87 … 0.52 |
| waist_yaw | ±88 | ±2.618 |
| **waist_roll / waist_pitch** | **±50** | **±0.52（仅 ±30°）** |
| shoulder_pitch / roll / yaw | **±25** | pitch −3.09 … 2.67 |
| elbow | **±25** | −1.05 … 2.09 |
| wrist_roll | 25（mjlab；Menagerie 未标注） | ±1.972 |
| **wrist_pitch / wrist_yaw** | **5** | ±1.614 |
| Dex3 `thumb_0` | **±2.45** | ±1.047 |
| Dex3 `thumb_1/2`, `index_0/1`, `middle_0/1` | **±1.4** | 0 … 1.75 |

两个必须记住的"硬约束"：
- **腰部 pitch/roll 只有 ±30°**。摆动的泵能不靠大幅度折腰，必须靠"小幅度、对相位"的参数泵能（parametric pumping），这一点要写进设计与预期。
- **腕 pitch/yaw 只有 5 N·m**，在悬垂载荷路径上基本不能承力 —— 这是 §3 里把它们锁死的直接理由。

### 2.2 力矩预算：悬垂到底行不行（关键分析）

整机重力 $W = 32.24\times9.81 \approx 316\ \mathrm{N}$。

**(a) 双臂同杆悬垂**：每臂 158 N。若臂竖直（载荷线穿过肩/肘/腕回转轴），静力矩 ≈ 0，关节只需承担摆动惯性矩：

$$
\tau_{\text{swing}} \sim I\ddot\theta,\qquad I \approx mL^2 \approx 32 \times 0.5^2 \approx 8\ \mathrm{kg\,m^2}
$$

结论：**"直臂悬垂 + 摆动"在 25 N·m 下是可行的**；但静态把身体举离铅垂线的能力很有限：

$$
\tau_{\text{shoulder}} = F\cdot x \Rightarrow x_{\max} = \frac{25}{158}\approx 0.16\ \mathrm{m}\ (\text{双手}),\qquad \frac{25}{316}\approx 0.079\ \mathrm{m}\ (\text{单手})
$$

**(b) 单手悬垂（brachiation 的关键相）**：单臂 316 N，只要质心偏离铅垂线超过 **8 cm**，肩关节就饱和。含义非常重要：

> **G1 不可能"静态地"单手挂住一个偏置的身体；它必须像真正的吊环/云梯运动员那样，用近直臂 + 摆动通过。**
> 这不是缺点，反而使"必须靠摆动与动量"成为物理事实，而不是我们人为设的难点。

**(c) 腰部的角色**：腰部要控制 17.9 kg 的下半身，力臂 0.36 m：

$$
\tau_{\text{waist}} = 17.9\times9.81\times0.36\times\sin\varphi \approx 63\sin\varphi\ \mathrm{N\cdot m}
$$

50 N·m 对应 $\varphi\approx 52°$ 的静态保持能力 —— 而行程只有 ±30°，所以**腰部可以"轻松地"在行程内驱动整条下半身摆动**。这是全机最有效的泵能执行器，也是 §3 把 waist_pitch 放进动作空间的核心理由。

**(d) 抓握：最大风险**。双手悬垂时每手指约 $158/3\approx 53$ N，指节回转轴到杆心 ≈0.04 m：

$$
\tau_{\text{finger}} \approx 53 \times 0.04 \approx 2.1\ \mathrm{N\cdot m} \quad \text{vs 上限 } 1.4\ \mathrm{N\cdot m}
$$

单手悬垂时约 4.2 N·m，**明确超出**。这是**数量级估算**，真实值取决于手指包络几何（若杆被"钩"在掌指关节附近、载荷走指节连杆而非关节力矩，需求会下降）。但方向是明确的：

> **原装 Dex3-1 的手指不适合承担整机重量。** 抓握必须靠 (i) 钩挂式形封闭几何、(ii) 仿真中提高手指力矩上限（明确记为与真机的偏差）、或 (iii) 定制 1-DoF 夹爪（§3.4）。**M1 必须先做实测**。

### 2.3 由力矩预算直接导出的设计约束

1. 悬挂构型必须是"**近直臂**"：肘部接近伸直、肩部接近中立，载荷线穿过关节。这应当是 reset 的初态，也是策略应当收敛到的姿态。
2. 前进动力来自**摆动**（动量），不是静态支撑。因此任务的关键技能是"泵能与相位"，与 `思路.md` 的"摆动—释放—前伸—抓握"完全一致。
3. 腕 pitch/yaw（5 N·m）不能放在载荷路径的控制目标上 → 锁死（§3.3）。
4. 抓握能力必须先测量、再决定末端方案，且**必须在 CRL 之前**（否则会在错误物理上浪费数周）。

---

## 3. 关节控制分配（本计划的核心之二）

### 3.1 六类分配语义（建议全项目统一用这套词）

| 类别 | 语义 | MuJoCo 实现 | 是否进动作空间 |
|---|---|---|---|
| **A 主动控制** | policy 直接给目标 | `<position>` + 动作缩放 | ✅ |
| **B 定值伺服** | 有 actuator，但目标固定，不随 policy 变 | `<position>` 目标 = 常数 | ❌ |
| **C 等式锁定** | 去掉该自由度，等价于刚性支架/机械锁 | `<equality><joint .../>`（MJX-JAX 支持 JOINT/WELD） | ❌ |
| **D 自由被动** | 无 actuator，仅关节阻尼/摩擦/限位 | 删掉 actuator，保留 `<joint damping frictionloss>` | ❌ |
| **E 弹性被动** | 被动 + 弹簧/软限位（"改造"） | `<joint stiffness springref>` 或 tendon spring | ❌ |
| **F 耦合欠驱动** | 多关节由少数 actuator 通过 tendon/synergy 驱动 | `<tendon>` + `<fixed>`/`<pulley>` 联动，或控制器层 synergy | 少维 ✅ |

一句方法论上的说明（写进论文也站得住）：
> We intentionally restrict actuation to a low-dimensional, deliberately underactuated action space so that the learning problem is about dynamic contact switching, not about joint-space exploration or dexterous grasping.

### 3.2 逐关节分配表（v1 主配置）

| 关节组 | 数量 | 类别 | v1 处置 | 理由 |
|---|---|---|---|---|
| 浮动基座（6 DoF） | 6 | D | **自由**（不可驱动） | 摆锤自由度；欠驱动的根源 |
| 腿：hip_pitch / roll / yaw / knee / ankle_pitch / roll | 12 | **B**（定值伺服，抱膝 tuck 姿态） | 保持固定目标；不是动作 | 无地面、不参与接触；抱膝缩短摆长、抬高质心、减小转动惯量（更接近人在云梯上的姿态）。**保留为消融轴**：改为 D（被动摆腿）或 A（1–2 维主动摆腿） |
| 腰：waist_yaw | 1 | **A** | 进动作 | 88 N·m；侧向/扭转调整与换手时的身体对位 |
| 腰：waist_pitch | 1 | **A** | 进动作 | 50 N·m vs 需求 63 sinφ；**主泵能执行器** |
| 腰：waist_roll | 1 | **B** | 固定 0 | 避免与 bar/大腿的自碰撞，减少一维探索；消融轴 |
| 臂：shoulder_pitch / roll / yaw | 6 | **A** | 进动作 | 主摆动与伸展；25 N·m |
| 臂：elbow | 2 | **A** | 进动作 | 调节摆长（parametric pumping 的第二个手段）+ 前伸抓取 |
| 臂：wrist_roll | 2 | **A** | 进动作 | 唯一必要的腕自由度：把 power grasp 的抓握轴转到与杆平行 |
| 臂：**wrist_pitch / wrist_yaw** | 4 | **C** | **等式锁定**（若有需要再降级为 B） | 仅 5 N·m，处于载荷路径上，动态中只会带来抖振与穿模；锁死等价于"加装刚性腕部支架" |
| 手：Dex3 7 关节/手 | 14 | **F** | **1 DoF/手 synergy**（开合），连续 $g\in[-1,1]$ | 研究问题是"何时放/何时抓"，不是"抓多好"。synergy 用固定基向量投影 $\mathbb R^7\to\mathbb R^1$（见 §3.4） |

**v1 动作向量（14 维，顺序即代码里的顺序）**：

```text
a = [ L_sh_pitch, L_sh_roll, L_sh_yaw, L_elbow, L_wrist_roll,
      R_sh_pitch, R_sh_roll, R_sh_yaw, R_elbow, R_wrist_roll,
      waist_yaw, waist_pitch,
      g_L, g_R ]
```

- 关节动作：`q_des = q_default + scale * a`，`a ∈ [-1,1]`，`scale = 0.25 * effort_limit / stiffness`（沿用 mjlab `G1_ACTION_SCALE` 的口径），并**按悬垂区间重新整定 PD**（Menagerie 的 kp=500 与 `g1_mjx` 的 kp=75/20 都是为站立/行走调的，不能直接用于悬挂）。
- 抓握动作：`c = (g+1)/2`，再做一阶低通（action smoothing）→ `z`；避免接触瞬间的抖振。**不做 binary 化**（保留连续，binary 作为消融）。
- 明确**不使用** `weld` 之类的"魔法吸附"：抓握成功必须是摩擦/接触的物理结果（`末端执行器设计.md` 的核心主张）。

### 3.3 为什么锁的是腕 pitch/yaw 而不是别的

- 悬挂载荷路径：杆 → 手指 → 掌 → **腕** → 前臂 → 肘 → 上臂 → 肩 → 躯干。腕 pitch/yaw 在这条路径上且只有 5 N·m，一旦被动态载荷激励就会饱和并被"压着走"，表现为穿模/抖振。
- 腕 roll 是 5020（25 N·m），且是把抓握轴与杆对齐的必要自由度，必须保留。
- 这不是"简化机器人"，是**保留原机结构、只对控制通道做取舍** —— 在论文里可写成 control-authority allocation，并作为消融轴。

### 3.4 末端执行器方案（三选一 + 一个消融）

**方案 1（v1 默认）：Dex3-1 + 控制器级 synergy（虚拟欠驱动）**

$$
q^{\text{des}}_{\text{hand}} = q_{\text{open}} + c\,v_{\text{curl}},\qquad c\in[0,1]
$$

- $v_{\text{curl}}$ 覆盖 7 个手指关节（index/middle 的 0/1 + thumb 0/1/2），四指组先包、拇指稍后，用不同的增益实现"先钩后扣"。
- 用**软 impedance**（低 kp + 足量阻尼）而不是刚性轨迹：碰到杆的手指自然停住，其余继续包络 —— 这就是 `末端执行器设计.md` 的"虚拟欠驱动"，无需真建 tendon。
- 优点：不改形态、资产现成、保留"以后加维度"的研究轴（1 → 2 → 全 7 DoF）。
- 风险：**指力不足（§2.2d）**，抓杆可能"看着像抓住、一动就滑开"。

**方案 2（保真/可控难度）：定制 1-DoF 夹爪（"改造为欠驱动"）**

- 几何：拇指 + 四指组两个 curved jaw（`思路.md` 的图），单自由度 $a_g\in[0,1]$，绕圆柱的形封闭。
- 关键收益：**闭合力矩由我们自己定**（建议 ±8 … 15 N·m @ 夹爪枢轴），从而把"握力"从研究变量里移除；同时保留"何时开、何时合"这一真正的研究变量。
- 也可以做成真正的欠驱动：一个 actuator + 差动/腱，多个指节自适应包络（MJX-JAX 支持 fixed/spatial tendon 与 JOINT/SITE/PULLEY 缠绕）。
- 代价：约 3–5 天建模 + 碰撞代理 + 手掌安装标定。

**方案 3（仅作消融，不做主配置）：passive hook**
- 直接复现 ETH 的设定，用来隔离"主动松手"这一变量的贡献。它就是 `思路.md` 里明确要避开的做法，但作为 ablation 非常有价值。

**M1 的决策规则**（写死，避免反复）：
1. 若 Dex3 synergy 能通过"单手悬垂 ≥ 1 个完整摆动周期不脱落"（脚本化，非学习），→ 用方案 1。
2. 否则若把手指力矩上限 ×4 后能通过，而你不介意记录该偏差 → 方案 1 + 记录偏差，并同时报告方案 2 的结果。
3. 否则 → 方案 2。

### 3.5 动作空间消融矩阵（研究轴，也是论文的"机制分析"章）

| ID | 变更 | 想回答的问题 |
|---|---|---|
| A0 | v1：14 维（10 臂 + 2 腰 + 2 grasp） | 基线 |
| A1 | +waist_roll（15） | 侧向调整是否帮助换手？ |
| A2 | 解锁 wrist pitch/yaw（18） | 更自由的腕是否带来新抓法，还是只是抖动？ |
| A3 | 腿：B→D（被动摆腿，仍 14 维） | 被动欠驱动腿是否被利用为泵能？ |
| A4 | 腿：+2 维主动 hip_pitch（16） | 主动摆腿是否显著降低探索难度？（"人是怎么做的"对照） |
| A5 | 抓握 1 DoF → 2 DoF（四指 curl + 拇指 opposition，16） | grasp-control dimensionality 研究轴 |
| A6 | 连续抓握 → 二值开合 | 半闭合接触技巧是否存在？ |
| A7 | 方案 1 → 方案 3（hook） | 主动释放的价值 |
| A8 | 单侧手不可释放（永久 weld 一只手） | 必须"主动放弃稳定接触"这个命题的强度 |
| A9 | $d$ ∈ {0.25, 0.30, 0.40}、杆半径 ∈ {0.02, 0.03} | 难度-可达性的相变点 |

---

## 4. 软件栈评估与选型

### 4.1 候选对比

| 方案 | CRL 现成度 | 接触性能 | G1 资产 | 工程风险 | 结论 |
|---|---|---|---|---|---|
| **JaxGCRL + MJX-JAX** | ★★★★★（CRL/SAC/TD3/HER 全有） | ★★★ | 需自建 43-DoF 的 MJX 场景 | 低 | **主路线** |
| MuJoCo Playground + MJX-Warp + 自研 CRL | ★★（CRL 要自己写，约 200–300 行） | ★★★★★（官方实测 Warp 数 M SPS） | G1 官方已适配 MJX | 中 | **降级路线**（吞吐不达标时） |
| mjlab + MuJoCo Warp + 自研 CRL | ★★ | ★★★★★ | 本地已有 `my-mjlab`，含真实电机模型 | 中高（要接 Warp 数据做 JAX 训练） | 备选 |
| 原 SGCRL | ★★★★★ | ★★★ | — | 极高（依赖腐烂） | 排除 |

**为什么仍以 JaxGCRL 为主**：它把 CRL 与 SAC+HER / TD3+HER 基线放在同一个训练框架里，"是 CRL 特别有效还是任何 GCRL 都行"这个审稿人必问的问题才有低成本答案；而且官方数据点显示 1000 万步 ≈ 10 分钟/单卡。

**实测确认的 JaxGCRL 事实**（已核对仓库源码与 README，Apache-2.0）：
- 自定义环境的接口是 **brax `PipelineEnv`**：`sys = brax.io.mjcf.load(xml_path)`，`super().__init__(sys=sys, backend="mjx", n_frames=...)`；`reset/step` 必须 JIT-able。
- 加新环境 = ① 把 XML 放到 `jaxgcrl/envs/assets/`；② 写 `jaxgcrl/envs/xxx.py`（照 `envs/humanoid.py` 改）；③ 在 `utils/env.py` 的 `legal_envs` / `create_env` 注册。
- **已有 `humanoid` 与 `humanoid_{u,big,hardest}_maze`** —— 说明"带接触的三维人形"在这套栈里已经跑通过，本任务不是第一个吃螃蟹的。
- 基线：CRL / PPO / SAC / SAC+HER / TD3 / TD3+HER；CLI `jaxgcrl crl --env ... --num_envs ... --total_env_steps ... --use_her`。
- 默认配置：`total_env_steps=5e7`、`episode_length=1001`、`num_envs=256`、`action_repeat=1`、`backend ∈ {mjx, spring, positional, generalized}`。
- **本项目必须 `backend="mjx"`**，不要用 `spring/generalized`（后者是 Brax 的简化物理，抓握/摩擦不可信）。

### 4.2 MJX 能力核对（逐条对照官方文档）

| 特性 | MJX-JAX | MJX-Warp | 对本项目的含义 |
|---|---|---|---|
| capsule / sphere / box / mesh / plane | 全支持（mesh 用 SAT） | 全支持 | **杆用 capsule**；手指碰撞用 capsule/sphere/box 代理 |
| cylinder / ellipsoid | 只能与图元碰撞 | 全支持 | 不要用 cylinder 做杆 |
| 大 mesh 碰撞 | SAT，性能差（mesh-primitive <200 顶点，convex-convex <32） | 好 | **G1 的 mesh 碰撞必须换成图元代理**（`g1_mjx.xml` 已经这么做了：27 个 capsule collider） |
| equality（CONNECT/WELD/JOINT/TENDON） | 支持 | 支持 | §3.3 的"锁腕"可以直接用 `<equality><joint>` |
| tendon（fixed/spatial + JOINT/SITE/PULLEY 缠绕） | 支持 | 支持 | 方案 2 的"真欠驱动夹爪"在 MJX 下可行 |
| position actuator | 支持 | 支持 | 位置伺服控制契约可行 |
| 显式接触对 (`<pair>`) | **强烈建议** | 建议 | **`g1_mjx.xml` 的 contact 对只覆盖了脚-地**，本项目要自己写全套（手指↔杆、手↔躯干、腿↔腿…），并尽可能稀疏 |
| autodiff | 部分支持 | **不支持** | CRL 不需要，无影响 |
| 并行吞吐 | 好（但接触多时下降明显） | 最好 | 决定了 §4.3 的吞吐门限 |

### 4.3 推荐路线 + 降级触发条件

```text
Menagerie G1(43) ──► MJCF 改造（图元碰撞+显式接触对+杆+锁腕+手代理）
                          │
                          ▼
                 普通 MuJoCo + Viewer 单环境（CPU）
                 ├─ 悬挂/摆动/释放-重抓 脚本化验证   ← M1 硬门限
                 └─ 接触对数、穿透、抖振、PD 整定
                          │
                          ▼
              JaxGCRL + backend="mjx"（自定义 braxiation env）
                          │
                 M0.5 吞吐门限：256 envs 稳定跑通且
                 ≥ 5k env-steps/s ？ ──否──► 降级：MJX-Warp
                          │是                        （brax 的 mjx pipeline 已显式
                          ▼                            处理 data._impl.contact，
                16→64→256→1024 envs（本地 4060）        值得先做 1 天 spike）
                          │
                          ▼
                服务器 GPU：1024–8192 envs
```

**降级路线的可行性备注**：brax `brax/mjx/pipeline.py` 里已经出现 `data._impl.contact if hasattr(data, '_impl')` 这种分支，说明上游已经在考虑 MJX-Warp 的接触缓冲差异；因此"JaxGCRL 直接用 `impl='warp'` 的模型"值得用一个 1 天的 spike 验证（`mjx.put_model(mj_model, impl='warp')` + `make_data(naconmax=..., njmax=...)`）。若不通过，就退到 MuJoCo Playground 的环境 API + 自研 CRL（CRL 损失本身只有约 20 行）。

### 4.4 工程结构（建议）

```text
monkeyBars_CRL/
├── docs/                          # 本文档与既有三篇
├── assets/
│   └── g1_brachiation/
│       ├── g1_43dof_mjx.xml       # 由 g1_with_hands.xml 改造：图元碰撞+PD+锁腕
│       ├── grip_proxies.xml       # Dex3 手指的 capsule/sphere 碰撞代理
│       └── scene_bars.xml         # 5 根 capsule 杆 + 显式 <pair> + hang keyframe
├── configs/
│   └── joint_allocation.yaml      # §3.2 的机器可读版（见附录 B）
├── src/
│   ├── envs/brachiation.py        # brax PipelineEnv 子类（obs/goal/terminate）
│   ├── control/synergy.py         # Dex3 1–2 DoF synergy 映射 + 低通
│   ├── eval/metrics.py            # P(max bar ≥ k)、逗留成功率、接触切换统计
│   └── tools/
│       ├── hang_ik.py             # 求解双臂同杆悬挂 keyframe
│       ├── torque_budget.py       # §2.2 的自动核算（给定构型输出各关节 τ）
│       └── replay_skill_timeline.py
└── runs/                          # 训练产物（gitignore）
```

---

## 5. 风险评估

| ID | 风险 | 概率 | 影响 | 触发信号 | 缓解 / 决策点 |
|---|---|---|---|---|---|
| **R1** | **Dex3 指力不足，抓握不成立** | 高 | 致命 | 静态悬垂时手指被"撑开"、滑脱 | M1 实测；备选：钩挂几何 / 提高指力矩上限（记为偏差）/ 定制 1-DoF 夹爪（§3.4） |
| **R2** | 单手悬垂在 25 N·m 下不可行，导致 release→reach 根本做不到 | 中 | 高 | 脚本化单手悬挂会立刻绕肩"折"下去 | 直臂悬垂构型 + 摆动通过；把"允许的关节力矩"作为一条明确的偏差轴报告 |
| **R3** | 接触参数（condim/friction/solref/solimp）导致抓握抖动/穿透 | 中 | 中 | viewer 里可见抖振、接触力尖峰 | condim=4（含扭转摩擦）、显式 pair 调 friction、减小 timestep、手指碰撞用图元 |
| **R4** | **探索失败**：14 维动作 + 接触切换 + 无 reward，连 B0→B1 都出不来 | 中高 | 中（本身是可发表结论） | 5e7 步内 $P(\text{reach }B_1)\approx 0$ | 分阶段目标（先 B1）；易化 $d$；A4 消融（主动摆腿）；HER 对照；把"探索边界"作为产出写进论文 |
| **R5** | 退化解："扑到 B4 附近然后掉" | 中 | 中 | success 高但逗留成功率低 | §1.4 的 k 步逗留判据 + 目标占据率并列报告 |
| **R6** | MJX-JAX 在 G1+多接触场景吞吐不足 | 中 | 中 | 256 envs < 5k env-steps/s | M0.5 门限 → MJX-Warp / Playground |
| **R7** | 自建 43-DoF MXJ 场景的工程量被低估（碰撞代理 + 接触对 + 悬挂 keyframe） | 中 | 中 | M0 超期 | 用 `g1_mjx.xml` 当模板（碰撞几何与 solver 都已调好）；腿/头的接触对可以整体砍掉 |
| **R8** | goal 设计不当（信息过多/过少）导致学不动或学到捷径 | 中 | 中 | 训练曲线 $P(\max\ge k)$ 长期为零，或 success 高但视频里是"甩过去" | §1.4 的 goal 集合 + 消融（去掉 contact one-hot） |
| **R9** | 本机当前无 GPU（`nvidia-smi` 失败、JAX 只见 CPU），本地无法训练 | 中 | 中 | — | M0/M1 全部在 CPU + 普通 MuJoCo 上完成（本来就应该这样）；训练前确认宿主 GPU 与 CUDA≥12.3 |
| **R10** | 论文叙事被"用了完整 G1 但改了指力矩/腕锁"削弱 | 低中 | 低 | 审稿人质疑保真度 | 每项改造都作为**显式偏差表**报告，并给出"不改也能学到什么"的消融（A2/A7） |

---

## 6. 执行计划（里程碑）

每个里程碑都有**硬验收标准**与**失败分支**，不允许"看起来差不多就往前走"。

> **执行状态（已更新）**：**M0 完成（✅）、M1 通过（✅，验收口径按 §M1 调整）、M0.5 / M2 进行中**。
> M1 的验收口径经确认调整为：**只要证明"双手能挂住"且"单手也能独立支撑身体挂住"即可**；"松开一只手的时机与再抓的协同"属于策略要学的技能，不作为物理门限（M1.3 记录的"松手真实、IK 前伸可达并接触、但脚本化闭合未能维持"保留为一条观察，不再作为验收项）。
> 当前配置：**正手抓杆**（手掌与脸同向，`palm·(+x)=+1.00`）、指力上限 ×10。细节见 **`docs/M0_M1_报告.md`**；手部命令细节见 **`docs/手部控制说明.md`**；结果在 `runs/m1/`。
> 影响后续计划的实测结论：① 原装 Dex3 指力 ×1 不够，正手姿态下需要 **×10**；② 正手（手掌朝前）力学上明显优于反手——手落在肩前方，质心可精确压到杆下，臂部力矩利用率从 0.96 降到 **0.39**；③ 单手悬垂受肘 25 N·m 限制，必须近直臂；④ 执行计划 §4.2 建议的"稀疏接触图"与 Dex3 抓握冲突（手指会 curl 过头把杆放掉），M2 需重新设计。

### M0 — 环境与资产生成（3–5 天）✅ 已完成
- 建 `assets/g1_brachiation/`：以 `g1_with_hands.xml` 为形态基座、以 `g1_mjx.xml` 为 MJX 改造模板，产出 43-DoF 的 MJX 就绪场景（图元碰撞代理 + 显式接触对 + 5 根 capsule 杆 + 锁腕 equality）。
- 写 `hang_ik.py`：解出双手同抓 B0 的对称悬挂 keyframe（直臂、腰中立）。
- 写 `torque_budget.py`：对给定构型自动输出各关节重力/惯性力矩，与 §2.2 的手算对表。
- **验收**：① 普通 MuJoCo viewer 里 G1 双手挂 B0 静置 10 s 不掉、不穿模；② 接触对总数与每步接触点数有记录（用于 MJX 调参）；③ 手动施加肩/腰正弦指令能看到明显摆动。
- **失败分支**：若悬挂构型无法同时满足双手抓握（IK 无解 / 手臂行程不够），把杆半径或杆高度作为自由参数重新求解，并记录可用的几何区间。
- **实际结果**：`build_scene.py`（场景生成 + 姿态求解 + 关键帧）+ `src/torque_budget.py`；正手配置 60 s 静置漂移 **5 mm**。**与原计划的差异**：不用"直臂、腰中立"而是用 `hip_pitch≈1.3 rad` 的腿后摆把质心压到杆下（G1 过头时手在肩后 8–13 cm）；关键帧必须由"零重力合手 → 重力斜坡"生成，不能靠运动学摆放（否则 24–38 mm 穿透把机器人弹飞）。

### M0.5 — 训练栈打通与吞吐门限 🔄 仅剩"在 GPU 上确认"这一步

> **最新状态（以 `docs/M2_JaxGCRL接入记录.md` §8 为准）**：
> - ✅ 后端选定并**实测可用**：MJX-Warp 能跑**全保真 mesh 模型**（无需碰撞代理），本机 CPU 也能验证物理（悬挂漂移 −5.6 mm vs 原生 −5.2 mm）。
> - ✅ 环境 adapter（方案 A：物理走 `mjx.step`，brax 仅作类型库）已实现并自检通过。
> - ✅ 损失函数可配置（dot/L2 × InfoNCE/binary 的 2×2 + bias/scale），JaxGCRL 的最小补丁存在 `patches/jaxgcrl_crl_losses.patch`。
> - ✅ 两个集成坑已修：AutoResetWrapper 对 `mjx.Data` 空/非 batch 叶子；`goal_indices` 必须是 Python int 元组。
> - ⏳ **本机 CPU 编译完整训练图会 OOM**，"跑通训练"必须在 4060 上确认。

#### 以下为原计划文本（保留参考）
- **已完成**：JaxGCRL 的 pin 环境装好（`.venv-rl`：jax 0.4.25 / brax 0.12.1 / mujoco 3.2.7）；场景改为"导出静态 MJCF"以解耦两个 MuJoCo 版本；M1 三项验收已在 3.2.7 上复现（漂移 5.5 mm），证明物理跨版本一致。
- **已排除的 MJX 障碍**：cylinder↔mesh 碰撞（圆柱→capsule）；86 个碰撞 mesh 导致 OOM（改用 capsule 代理 + 剥视觉 geom）。
- **剩余障碍**：mesh 手指仍让 XLA 编译 OOM；全图元版本能在 MJX 编译通过，但 AABB 拟合的手指 capsule 太粗、抓握标定不出来 → 需要专门设计手指碰撞代理，或改走 MJX-Warp（支持全 mesh，但需 GPU）。详见 **`docs/M2_JaxGCRL接入记录.md`**。
- **吞吐门限仍需 GPU 机器**：本容器无 GPU，`pipeline.init` 300+ s、jit 编译 150–330 s 均为 CPU 数字，不能用于判断。
- 装 JaxGCRL（`pip install -e .` + CUDA），跑通 `jaxgcrl crl --env ant` 与 `humanoid`（确认基线栈可用）。
- 写最小的 `Brachiation(PipelineEnv)` 骨架：注册进 `utils/env.py`，只做 `mjcf.load(xml)` + 随机动作 step + jit。
- 测 16/64/256 envs 的 env-steps/s；顺带做 1 天 MJX-Warp spike。
- **验收**：256 envs 下稳定跑通且 **≥5k env-steps/s**（低于则触发 R6 降级）；`reset/step` 全 JIT 无 fallback。
- **失败分支**：走 §4.3 的降级路线。

### M1 — 物理保真与握力可行性（1–2 周，**项目最重要的门限**）✅ 通过
脚本化（非学习）验证四件事：
1. 双手悬垂静置：稳定，关节力矩 < 上限。→ **PASS**（60 s，双手抓握 100%，z 漂移 13 mm）
2. **单手悬垂**：能否维持 ≥1 个完整摆动周期（这是 brachiation 的必要条件）。→ **PASS**（4 s，但肘关节 100% 利用率，余量很小）
3. **release → reach → regrasp**：先在 B0 上做"松开一只手 → 再抓回同一根杆"。→ **不作为验收项**（经确认）：松手真实、IK 前伸可达并接触已证明；闭合后的协同与时机属于策略学习内容。观察记录见 `M0_M1_报告.md` §5
4. 摆动泵能：仅用 waist_pitch/shoulder 的正弦指令，能否把摆幅从 0 泵到 ±20°。→ **PASS**（改用髋/膝 0.55 rad @0.6 Hz：脚端 **280 mm**、质心 93 mm 峰峰；实测腰部同相激励反而减小摆幅）
- **验收（调整后）**：① **双手悬垂**稳定 + ② **单手悬垂**能独立支撑身体（≥8 s）→ 通过，进入 M2。③ 的"再抓协同"由策略学习，不作物理门限。
- **产出**：偏差表（改了什么、为什么、影响）、力矩预算实测表、以及与腕锁/夹爪方案对应的对照视频。→ 全部在 `docs/M0_M1_报告.md` + `runs/m1/`

### M2 — 环境接口与 CRL 接入（3–5 天）🔄 进行中
- **变量定义提案已出**：`docs/M2_任务变量定义.md`（动作 14 维、状态 111–125 维、goal 四套方案 G1/G2/G2b/G3 及推荐、终止/成功/指标、reset 分布、JaxGCRL 落地字段），等待确认 Q1–Q7。
- **关键机制发现**：JaxGCRL 的 CRL 在训练期把 goal **从未来状态重标记**（`goal = future_obs[:, goal_indices]`），所以"自生成课程"是自动发生的，不需要课程代码；`goal_indices` 的选择等于决定课程的粒度。
- 定稿观测/动作/goal/终止：obs ≈ 122 维（$q,\dot q$ 86 + root 位姿 3+3 + 双手位置 6 + 抓握 one-hot 10 + 上一步动作 14），goal 见 §1.4；写 `goal_indices` 与 `state_dim`。
- 接通 CRL：无 reward 路径、目标采样、replay buffer、CSV/wandb 日志、评估器（$P(\max\ge k)$、逗留成功率）。
- `action_repeat` 与 `timestep`：物理 timestep 0.004→0.002 量级，env step = 4–5 个物理步（与 `g1_mjx` 口径一致）。
- **验收**：单 seed、5e6 步无 NaN、吞吐稳定、评估脚本能输出 §1.5 的指标；在一个玩具任务（把"摆幅最大化"或"单手保持"当 goal）上能学到东西，证明栈本身没有 bug。
- **失败分支**：若连玩具 goal 都学不动，问题在栈/接口而不在任务，先修栈。

### M3 — 单杆跨越（B0→B1）与首次技能涌现（1–2 周）
- goal = 抓 B1；先跑易化 $d=0.30$，再 0.40。
- 记录：首次单手释放、首次触碰 B1、首次抓住 B1 的时间；绘制 $P(\max\ge1)$。
- **验收**：至少出现"偶发松手但不掉"（$\mathcal M_{LR}\to\mathcal M_L$）与"偶发触到 B1"这两种中间行为；若能稳定抓到 B1 即为强结果。
- **失败分支**：无任何中间行为 → 依次启用 A4（主动摆腿）、A9（$d=0.25$）、HER 对照；若全部无效，转向"探索边界"的负面结果叙事。

### M4 — 多杆连续前进（2–4 周）
- $d=0.40$、5 根杆、goal = B4；goal 是否用集合见 §1.4。
- **验收**：$P(\max\ge 2)$ 明显非零、$P(\max\ge3)$ 出现；给出连续换手视频与接触模式时间线图。

### M5 — 机制分析与消融（2–3 周）
- 跑 §3.5 的 A1–A9（先做 A4/A5/A7/A8 这四个最有信息量的）。
- 基线对照：CRL vs SAC+HER vs TD3+HER（同一环境、同一预算、多种子）。
- **验收**：每个消融至少 3 个 seed，报告均值±方差与"技能出现时间"。

### M6 — 产出（1–2 周）
- 主图：$P(\max\ \text{bar}\ge k)$ 与技能涌现时间线；机制图：泵能/相位/接触力。
- 偏差表、失败实验记录、可复现脚本与资产。

---

## 7. 算力与预算

| 阶段 | 环境数 | 预估吞吐 | 步骤量 | 本机(4060 8G)墙钟 |
|---|---|---|---|---|
| M2 冒烟 | 16–64 | — | 5e6 | 分钟级 |
| M3 单杆 | 256 | 待测（门限 5k/s） | 1e7–3e7 | 1–2 h/seed |
| M4 多杆 | 256–1024 | 待测 | 5e7/seed × 3 seeds | 1–3 天 |
| M5 消融 | 256 | — | 9 组 × 3 seeds × 2e7 | 数天–1 周 |
| 最好情况复现 | 4096+ | 服务器 | 1e8–2e8 | 服务器 |

注：JaxGCRL 官方口径是"1000 万步/10 分钟/单卡"，但那是接触稀疏的 benchmark；本场景接触密集，**必须以 M0.5 的实测为准**，上表只是排期用。本机当前 `nvidia-smi` 不可用、JAX 只见 CPU（容器内），M0/M1 的 CPU 单环境开发不受影响，训练前需要确认宿主 GPU 与 CUDA ≥ 12.3。

---

## 8. 需要你拍板的开放决策

| # | 决策 | 我的建议 |
|---|---|---|
| D1 | 保真度取向：仿真研究优先，还是必须保留 sim2real 可能？ | 仿真优先 + 保留 sim2real 通道（力矩上限/几何不改，改动全部列表化） |
| D2 | 末端执行器：Dex3 synergy / 定制 1-DoF 夹爪 / hook | 先按 D2=Dex3 synergy 开工，**由 M1 的实测结果决定**，M1 前不定死 |
| D3 | 腿：定值抱膝 / 被动自由 / 1–2 维主动摆动 | v1 定值抱膝（14 维动作）；A3/A4 作消融 |
| D4 | 训练栈 | JaxGCRL + MJX-JAX 为主，M0.5 门限不达标即切 MJX-Warp |
| D5 | goal：单个 B4 终态 vs 目标集合（B1..B4） | v1 单目标 B4（对应研究命题）；目标集合作为对照实验 |
| D6 | reset 分布是否随机化（含起始杆索引） | 起始固定 B0（不要送课程）；仅对初态加微小噪声 |
| D7 | 腕 pitch/yaw 用等式锁死，还是仅定值伺服 | 锁死（有 5 N·m 载荷路径问题），A2 里解锁做对照 |

---

## 附录 A：关键数值与出处（可核查）

| 数值 | 来源 |
|---|---|
| G1 43 关节命名、行程、`actuatorfrcrange`（含 Dex3 `thumb_0 ±2.45`、其余 `±1.4`） | `mujoco_menagerie/unitree_g1/g1_with_hands.xml`；`unitreerobotics/unitree_ros` `g1_29dof_with_hand_rev_1_0.xml`（本项目已下载核对） |
| MJX 碰撞几何（capsule）、`<pair>` 列表、solver/PD 设置（kp=75/20, kv=2, timestep 0.004, iterations 5, ls 8, implicitfast） | `mujoco_menagerie/unitree_g1/g1_mjx.xml`、`scene_mjx.xml` |
| 全机质量 32.24 kg、分段质量、链长 | 直接解析 Menagerie MJCF 的 `<inertial>` 与 body 层次（本文 §2.1） |
| 电机型号与真实力矩/速度上限（5020=25、4010=5、7520-14=88、7520-22=139）、10 Hz 自然频率 PD 整定口径 | `mjlab/asset_zoo/robots/unitree_g1/g1_constants.py`（本地副本 `/home/firedust/Robot-Arm/my-mjlab`） |
| MJX 支持矩阵（geom/equality/tendon/actuator/condim、mesh 性能建议、显式接触对建议） | MuJoCo 官方 `doc/mjx.rst`（3.9.0） |
| MJX-Warp 与 MJX-JAX 分工、Warp 接触/约束性能改善、Warp 无 autodiff | 同上 |
| JaxGCRL 环境接口/注册方式/已有 humanoid 与 maze 环境/基线清单/默认超参 | `JaxGCRL` 仓库（master）：`README.md`、`jaxgcrl/envs/humanoid.py`、`jaxgcrl/utils/env.py`、`jaxgcrl/utils/config.py` |
| 完整 G1 的 MJX 化与真机迁移先例 | Menagerie `unitree_g1/README.md`（MuJoCo Playground, arXiv:2502.08844） |

## 附录 B：机器可读的关节分配（`configs/joint_allocation.yaml` 草案）

```yaml
robot: unitree_g1
source_assets:
  morphology: mujoco_menagerie/unitree_g1/g1_with_hands.xml   # 43 actuated joints
  mjx_template: mujoco_menagerie/unitree_g1/g1_mjx.xml        # capsule colliders + pairs
actuated_joints: 43
free_base: {dof: 6, class: D_free}

groups:
  legs:
    joints: ["{l,r}_hip_pitch", "{l,r}_hip_roll", "{l,r}_hip_yaw",
             "{l,r}_knee", "{l,r}_ankle_pitch", "{l,r}_ankle_roll"]
    class: B_fixed_servo
    target: tuck            # 抱膝；消融 A3 -> D_free, A4 -> A_policy(hip_pitch x2)
  waist:
    waist_yaw:   {class: A_policy}
    waist_pitch: {class: A_policy}
    waist_roll:  {class: B_fixed_servo, target: 0.0}   # 消融 A1 -> A_policy
  arms:
    policy: ["{l,r}_shoulder_pitch", "{l,r}_shoulder_roll", "{l,r}_shoulder_yaw",
             "{l,r}_elbow", "{l,r}_wrist_roll"]
    locked: ["{l,r}_wrist_pitch", "{l,r}_wrist_yaw"]   # class C_equality_lock；消融 A2 -> A_policy
  hands:
    class: F_coupled_underactuated
    synergy: 1            # v1: q_des = q_open + c * v_curl, c in [0,1]
    joints_per_hand: 7    # thumb_0/1/2, middle_0/1, index_0/1
    smoothing: first_order_lowpass
    # 消融 A5 -> synergy 2 (curl + thumb opposition)；A6 -> binary open/close
    # 若 M1 判定指力不足 -> 替换为 custom 1-DoF jaw（见 §3.4 方案 2）

action_space_v1:
  dim: 14
  order: [L_sh_pitch, L_sh_roll, L_sh_yaw, L_elbow, L_wrist_roll,
          R_sh_pitch, R_sh_roll, R_sh_yaw, R_elbow, R_wrist_roll,
          waist_yaw, waist_pitch, g_L, g_R]
  joint_scaling: "0.25 * effort_limit / stiffness"
  gripper_scaling: "q_des = q_open + ((g+1)/2) * (q_closed - q_open)"

scene_v1:
  bars: {count: 5, spacing: 0.40, radius: 0.025, type: capsule, axis: [0, 1, 0], z: 2.0}
  floor: none
  reset: {both_hands_on: bar0, qvel: 0}
  terminate: {root_z_below: 1.0, max_episode_steps: 1000}

goal_v1:
  fields: [root_x, root_z, grasp_L_onehot(5), grasp_R_onehot(5), root_vel]
  success: "at least one hand on bar_N for k=50 consecutive steps with root speed < eps"
```

---

# 9. 4060 上的首次运行（2026-09-17）—— 两个新问题与修复

用户机器实测：Warp 正常初始化并**同时看到 `cpu` 与 `cuda:0 (RTX 4060 Laptop, 8 GiB, sm_89)`**；环境/损失/配置/rollout 全部跑起来了（能看到物理步进），最后卡在评估器。

## 9.1 `KeyError: 'success_easy'`（已修）

`jaxgcrl/utils/evaluator.py` 里**硬编码**了一组指标名，会直接从 `episode_metrics` 里取：

```python
for name in ["reward", "success", "success_easy", "dist", "distance_from_origin"]:
    fn(eval_metrics.episode_metrics[name])
```

我们的环境没有提供 `reward` / `success_easy` / `distance_from_origin` ⇒ KeyError。修法两处：

1. **环境侧**（`src/envs/brachiation.py`）：补齐这 5 个名字，并额外加我们自己要的指标。
   * `reward` = 0（CRL 无 reward，但评估器要求这个键存在）；
   * `success_easy` = `dist < 3 × goal_reach_thresh`；
   * `distance_from_origin` = `‖qpos[:2]‖`；
   * `forward` = `qpos[0] − bar_x[0]`（**前向进度，头条图要用的量**）；
   * 原有的 `max_bar / dwell_success / hand_switches / bar_L / bar_R / fell` 保留。
2. **评估器侧**（`patches/jaxgcrl_crl_losses.patch`）：给列表加上我们的指标名，并加 `if name in eval_metrics.episode_metrics` 守卫 —— 这样缺哪个名字就跳过哪个，不再硬崩。

自检：`reset`/`step` 返回的 metrics 里 12 个键全部存在 ✓。

## 9.2 ⚠️ 最要紧的一条：`jaxlib` 是 CPU 版，所以刚才整轮跑在 CPU 上

运行日志里有一行很容易被忽略：

```
WARNING: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

Warp 用的是 CUDA，但 **JAX 用的是 CPU**（`.venv-warp` 里装的是 CPU 版 jaxlib）。CRL 的采样/训练都是 JAX 图，所以实际上没吃到 4060。解法：

```bash
uv pip install --python .venv-warp/bin/python -U "jax[cuda12]"
.venv-warp/bin/python -c "import jax; print(jax.devices())"     # 期望看到 [CudaDevice(id=0)]
```

装完再重跑 smoke。**在那之前所有吞吐数字都没有参考价值。**

## 9.3 `linesearch iterations limit reached` 刷屏（已抑制）

MJX-Warp 每命中一次 `ls_iterations` 上限就打印一个两行 block，几秒钟就刷了几百行。这是提示不是错误（不影响物理），已在
`src/mjx_backend.py::load_mjx` 里对 warp 模型设置 `opt.warn_overflow = 0` 抑制（`try/except` 包裹，失败也不影响运行）。
如果后续发现仿真质量受影响，再把 `option/ls_iterations` 从 20 提到 40–60（会轻微改变动力学，需要重跑 M1 复核）。

## 9.4 修完之后的下一步

```bash
# 1) 先确认 JAX 在 GPU 上
.venv-warp/bin/python -c "import jax; print(jax.devices())"

# 2) smoke（4k 步、4 envs）
.venv-warp/bin/python src/train.py --preset C_l2_infonce --smoke --impl warp --scene full

# 3) 真跑：上游原生默认组合（L2 能量 + forward InfoNCE）
.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
    --num-eval-envs 32 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
    --num-evals 20 --steps 4889600 --impl warp --scene full --wandb --exp-name brach_C
```

> 上面这段是**过时快照**（当时还有 `--preset D_l2_binary` 那一格；该预设与 bias/scale 一起已在 §9.15.1 删除）。
> 当前可用的预设只有 `C_l2_infonce`（默认）/ `A_dot_infonce` / `upstream_norm` / `S_l2_syminfonce`。

跑完把 `runs/<exp>/progress.csv` 贴回来，重点看 `eval/episode_max_bar`、`eval/episode_success`、
`eval/episode_dwell_success`、`eval/episode_forward` 和 `training/critic_loss`。

## 9.5 安装 CUDA 版 JAX（国内源）

`.venv-rl` / `.venv-warp` 最初装的都是 **CPU 版 jaxlib**（`jax.devices()` 只有 `CpuDevice`），所以 4060 上那轮其实跑在 CPU。
不要用 `uv pip install -U "jax[cuda12]"`：`-U` 会重解析依赖树并把 `numpy` 升到 2.x，而我们是整套链路在 `numpy==1.26.4` 上验证过的。
jax 0.6.2 的 `cuda12` extra 就等价于下面两个包，单独装即可，不动 numpy：

```bash
export UV_HTTP_TIMEOUT=300
uv pip install --python .venv-warp/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  "jax-cuda12-plugin==0.6.2" "jax-cuda12-pjrt==0.6.2"
```

（阿里源与清华源均已有 `cp310/x86_64` 轮子。清华：`https://pypi.tuna.tsinghua.edu.cn/simple/`）

验证与加速：

```bash
.venv-warp/bin/python -c "import jax; print(jax.default_backend(), jax.devices())"   # 期望 gpu [CudaDevice(id=0)]
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true      # 官方建议，约 +30%
```

注意：`np.bool = np.bool_` 这个 shim 在 numpy 2.x 下也成立（`np.bool_` 仍存在），所以即便将来升级 numpy 也不会立刻崩；
但**已验证的组合是 numpy 1.26.4**，没有明确理由不要动。

## 9.6 三处小修（4060 第二次运行）

| 报错 | 原因 | 修法 |
|---|---|---|
| `TypeError: progress() got an unexpected keyword argument 'do_render'` | JaxGCRL 的 `progress_fn` 契约是 `(num_steps, metrics, make_policy, params, env, do_render=...)`（见 `jaxgcrl/utils/env.py:265`），我们的回调只接了前两个 | `src/train.py` 的回调改成完整签名，忽略 `do_render`（不渲染视频） |
| `ls_iterations` 刷屏 | Warp 每命中一次上限就打两行 | `mjx_backend.silence_warp_overflow()`（`try/except` 包裹，**尽力而为**：本地无法验证 CUDA 路径下 `warn_overflow` 是否可写）；**零风险的替代做法**是把 stderr 过滤掉：<br>`... 2>&1 \| grep -v "linesearch iterations\|To disable the print warning"` |
| 本地沙箱看不到 GPU | 沙箱没有 nvidia runtime（`CUDA_ERROR_NO_DEVICE`），而用户 shell 里有 | 只影响我在容器内的验证；用户侧 `jax.default_backend()` 已是 `gpu` |

if the `warn_overflow` write turns out not to work on CUDA, the alternative is to raise `option/ls_iterations` from 20 to 40–60
in the exported scene — but that **changes the solver slightly and would require re-running the M1 acceptance**, so prefer the stderr filter.

## 9.7 ✅ 训练已在 4060 上跑起来（2026-09-17 23:42）

```
[env] mjx-warp scene=full action=14 obs=151 state_dim=141 n_frames=10 (50 Hz)
Module mujoco.mjx.warp.ffi ... load on device 'cuda:0' (cached)      ← Warp 跑在 GPU 上
  [     1440] episode_success=7.5  episode_dist=106.7  episode_max_bar=0  critic_loss=4.745  sps=50.24
  [     2640] episode_success=0.5  episode_dist=192    episode_max_bar=0  critic_loss=4.49   sps=234.3
  [     3840] episode_success=7.25 episode_dist=149.4  episode_max_bar=0  critic_loss=4.444  sps=227.6
```

**判读**：`critic_loss` 4.745 → 4.49 → 4.444 稳定下降 ⇒ 对比学习在学；`sps≈230`（4 个环境，smoke 规模）；
`max_bar=0` 是预期的（随机策略还不会前进）；`dist` 上百是因为策略掉下去后手/根节点离目标的 10 维欧氏距离很大。

### 9.7.1 最后一个报错：`assert total_steps >= config.total_env_steps`（上游 bug，已改为 warning）

JaxGCRL 只跑**整 epoch**（`num_envs × unroll_length`），epoch 数用整除取整，所以最终步数**系统性地达不到请求值**：

| 配置 | 请求 | 实际到达 | 差 |
|---|---|---|---|
| smoke（4 envs / 20 unroll / min_replay 50 / 3 evals） | 4 000 | 3 800 | −200 |
| JaxGCRL 默认（256 / 62 / 1000 / 200 evals） | 5×10⁷ | 4.787×10⁷ | −2.13×10⁶ |
| 我们计划的正式跑（256 / 62 / 1000 / 20 evals） | 5×10⁶ | 4.70×10⁶ | −3×10⁵ |

**连上游自己的默认配置都过不了这个断言**，所以它是 bug 而不是我们的错。已把 `assert` 改成 `logging.warning`（补丁同上）。
**`--steps` 应理解为"上限"**：实际步数 = `min_replay×num_envs + num_evals×⌊(steps−prefill)/(num_envs×unroll×num_evals)⌋×num_envs×unroll`。
想精确命中就先算一下：`steps_exact = prefill + k × num_evals × num_envs × unroll`。

## 9.8 `UnboundLocalError: params`（上游第二个 bug，已修）

`crl.py` 的 `train_fn` 里 `params` **只在 `if config.checkpoint_logdir:` 分支内被赋值**，所以不设 checkpoint 目录时会在
`return make_policy, params, metrics` 处炸。已在循环收尾处无条件导出最终参数：

```python
params = (training_state.alpha_state.params,
          training_state.actor_state.params,
          training_state.critic_state.params)
```

（补丁同上，现 142 行。顺带我们自己的入口也支持 `--checkpoint-dir` 保存最终参数。）

## 9.9 wandb 监测（已加）

`src/train.py` 新增：

```bash
.venv-warp/bin/python src/train.py --preset C_l2_infonce --smoke --impl warp --scene full --wandb
# 参数：--wandb-project（默认 monkeybars-crl）/ --wandb-group / --wandb-mode {online,offline,disabled}
```

* 首次使用需 `wandb login`；无网络时用 `--wandb-mode offline`（之后 `wandb sync` 补传）。
* 每次 eval 记录**全部** metrics，包括我们关心的：`eval/episode_max_bar`（头条量）、`eval/episode_success`、
  `eval/episode_dwell_success`、`eval/episode_forward`（前向进度）、`eval/episode_dist`、`eval/episode_hand_switches`、
  `eval/episode_bar_L/R`、`eval/episode_fell`，以及 `training/critic_loss`、`training/sps`、`training/log_alpha` 等。
* 同时仍然写本地 `runs/<exp>/progress.csv`（不依赖网络）。

## 9.10 smoke 通过：`[done] 71.1s`，wandb 正常

第一次完整跑完（`runs/brach_C_l2_infonce_warp/`，wandb run `tuzzejxr`）。summary 里两个看着奇怪但**不是 bug** 的数字：

* `eval/episode_hand_switches = 101` = episode 长度：随机策略下 `c=(a[12:]+1)/2` 平均 0.5（半开手），抓握指示器确实会逐步抖动。
  不过原来的 `hand_switches` 是**累积量**，经 eval 聚合器后语义含糊（求和会变成 1+2+…+n）。已改成**本步 0/1**，累积值移到 `info["switches_total"]`。
* `eval/episode_fell = 0` 与 `dist=149` 并存：**这条当时的解读是错的**（见 §9.18）。`eval/episode_*` 是"整条 episode 的**求和**"，所以 `fell=0` 就表示真的没掉（每步 0/1 求和），而 `dist=149` 只是 101 步的累加（≈1.5/步）；判读距离要除以 `avg_episode_length`。

## 9.11 两个会影响"可比性"的默认值（已改）

| 项 | 之前 | 现在 | 为什么 |
|---|---|---|---|
| `--episode-length` | 500 | **501** | JaxGCRL 硬检查 `num_envs*(episode_length-1) % batch_size == 0`。`500-1=499` 是奇数 → `num_envs` 必须是 512 的倍数；`501` 给 `500`，于是 `num_envs` 只要是 **128 的倍数**即可（128 envs / batch 512 已实测可跑） |
| 评估 instruction goal | 与训练一样**每 episode 随机** | 固定 `--eval-goal-bar 4`（默认 $B_4$） | 评估器自己 reset `eval_env`，随机 goal 会让 `eval/episode_success` / `episode_dist` 每次采到不同目标，**跨 run 不可比**。训练仍随机（`train_env.goal_bar=None`） |

> 注意：`eval/episode_success` 现在只反映"到 $B_4$ 那个 10 维悬挂态的距离是否 < 0.35"，比"随机 goal 的平均成功率"更严也更可读。要复现旧口径用 `--eval-goal-bar -1`。

顺带：训练 instruction goal 的采样下界做成了 flag `--train-goal-bar-min`（默认 0 = 均匀采 $\{B_0..B_4\}$）。
若首轮 5e6 步完全不前进，第一个要试的消融就是 `--train-goal-bar-min 1`（排除"起点即目标"的 $B_0$）。

## 9.12 终止条件 / 起始状态 / "是否一定挂得住"

三条一并定稿（细则见 `docs/M2_任务变量定义.md` §5–§6）：

1. **终止**：只有跌落 `root_z < bar_z - 0.8 = 0.686 m`。到达目标不终止（CRL 需要目标之后的未来帧，终止会抽干 self-curriculum 的正样本）。无其它 done，无地面/撞杆终止。
2. **起始状态**：关键帧 + 噪声（root ±5 mm、腿 ±0.02 rad、臂/手 ±0.01 rad、qvel ~ N(0,0.02)）→ **用关键帧 ctrl 空跑 10 步（0.02 s）settle** → 双手闭合、goal 随机采 $\{B_0..B_4\}$ 的标准悬挂帧。起点 = settle **之后**的状态。
3. **"是否一定挂得住"**：这是**实测问题，不是推导问题**（M1 已知抓握只容忍 1–2 cm 的手-杆偏差）。工具 = `src/check_reset.py`：固定噪声倍率 × $N$ 个并行 reset，再用**保持动作**（`a[0:12]=0, a[12:14]=+1`）空跑 2 s，输出 `grasp@reset`（必须 100 %，否则起点根本不是悬挂）、`held`（全程双手都在 $B_0$）、`alive`（且没掉过），并对噪声做 0→16× 扫描给出安全裕度。

```bash
# 在 4060 上跑（本容器无 GPU，跑不动；每个倍率会重新编译一次 jit，整体几分钟）
.venv-warp/bin/python src/check_reset.py
```

**结果待填**：跑完把输出贴进 `docs/M2_任务变量定义.md` §6.1 的表格下。

## 9.13 ⚠️ 一个会静默毁掉物理的坑：`naconmax` 是**全局**接触预算

第一次跑 `src/check_reset.py`（`--n 128`，full 场景）时 Warp 打出：

```
broadphase overflow - please increase nconmax beyond 8 or naconmax beyond 1024
```

含义：MJX-Warp 的 `naconmax` 是**所有 world 加起来**的接触容量（`mujoco_warp.make_data` 文档原话："Number of contacts to allocate for all worlds. Overrides nconmax"），
而 `nconmax` 才是每 world 的容量。原来是 `naconmax=1024`：

| 并行 world 数 | 每个 world 实际分到的接触容量 | 结果 |
|---|---|---|
| 4（smoke） | 1024/4 = 256 | 够用，之前的 smoke 是可信的 |
| **128（训练）** | **1024/128 = 8** | **溢出 → 多余的接触被丢弃** |

溢出不是"慢一点"，而是**丢掉接触**：抓握的接触可能被丢掉，"还挂着"就变得不可信。**所以之前 128 envs 的训练前实验必须在修好之后重跑。**

修法：
1. `naconmax` 随 batch 放大。`src/train.py` 新增 `--naconmax-per-world`（默认 128）→ `naconmax = per_world × num_envs`（128 envs → 16384，约几 MB，可忽略）；`--njmax`（每 world 约束行数）默认从 256 提到 512。
2. **之前那个"消除 Warp 刷屏"的补丁其实是个空操作**：`warn_overflow` 不在 MJX 的 `Option` 上，而在 Warp 的 `opt._impl` 上。`mjx_backend.silence_warp_overflow` 原来 `try: mx.opt.replace(warn_overflow=0) except: return mx` 一直走 except，所以警告照打——这也正是这次能发现溢出的原因。现改为
   `mx_model.replace(opt=opt.replace(_impl=opt._impl.replace(warn_overflow=0)))`（已在本机验证 `2047 → 0`）。
3. 因为警告被真正关掉了，**必须用数值去查**：`mjx_backend.overflow_bits(data)` 读取 Warp 的 `data._impl.overflow` 位掩码（`BROADPHASE=4`、`NARROWPHASE=8`、`CCD=16`…，0 = 没有丢接触），`src/check_reset.py` 把它做成表格里的 `ovf` 列。**`ovf != 0` 的那一行数据直接作废**，先加大 `naconmax`/`njmax` 再重跑。

附带一条 CCD 警告（不是错误，但要写进偏差表）：`MULTICCD is enabled, but the scene contains CCD pairs without multicontact support: [('CAPSULE','CYLINDER'), ('CAPSULE','MESH'), ('CYLINDER','CYLINDER'), ('CYLINDER','BOX'), ('CYLINDER','MESH')]. At most 1 contact will be generated for these pairs.`
即"杆 capsule ↔ 手指 mesh"每对最多产生 1 个接触点；抓握靠的是多根手指的多个接触对，所以仍有多点接触，但**接触数与原生 MuJoCo 不完全一致**，M1 的力矩预算结论要在 Warp 里复核一次。

另外 `src/check_reset.py` 自身有个 bug 已修：`env._grasp(data)` 只接受**单个**（未批量化）的 `mjx.Data`，直接喂 vmapped 的 `pipeline_state` 会因为 `data.xmat[bid]` 变成 `(nbody,3,3)` 而报 `cannot reshape ... into shape (3,3)`；现在改成 `jax.jit(jax.vmap(env._grasp))`。

### 9.13.1 修完之后的实测（2026-09-18，4060）

```
 scale grasp@reset    held   alive     z_min  dz_mean  fell   ovf    wall
   0.0      100.0%  100.0%  100.0%    0.7687  -0.0035     0  1024  185.8s
   1.0      100.0%  100.0%  100.0%    0.7678  -0.0033     0  1024   16.7s
   2.0      100.0%   98.4%   98.4%    0.6881  -0.0039     0  1024   18.0s
```

* **`ovf = 1024 = LS_ITERATIONS`（只有这一位）**，broadphase/narrowphase 都不再出现 ⇒ `naconmax` 放大到 `128×num_envs` 之后**接触没有再被丢弃**，之前的 broadphase 警告消失。剩下这一位是求解器线搜索触顶，不丢接触/约束，物理仍然是良态的；来源是场景 XML 的 `ls_iterations=20`（MuJoCo 默认 50，M1 也是用 20 验的）。想 A/B：`--ls-iterations 50`（`check_reset.py` / `train.py` 都已支持，env 侧参数是 `Brachiation(..., ls_iterations=)`）。
* **reset 容差**：1× 噪声 128/128 全挂住（`grasp@reset`/`held`/`alive` 全 100 %），2× 时 98.4 %（2 个滑脱，`z_min=0.6881` 离跌落阈值 0.6857 只剩 2.4 mm）⇒ **安全裕度约 2×**，当前默认噪声不需要改。
* `dz_mean ≈ −3.5 mm / 2 s`，与 M1 的 5 mm/60 s 同量级。
* 首次 185.8 s 是编译 + warp kernel 加载；之后每个倍率 ~17 s。

结论：**"随机初始化是否一定挂住"已实测回答：1× 下是（100 %），且容忍到约 2×。**

#### 9.13.2 全倍率扫描（0→16×，128 envs）

```
 scale grasp@reset    held   alive     z_min  dz_mean  fell   ovf
   0.0      100.0%  100.0%  100.0%    0.7687  -0.0035     0  1024
   0.5      100.0%  100.0%  100.0%    0.7683  -0.0034     0  1024
   1.0      100.0%  100.0%  100.0%    0.7678  -0.0033     0  1024
   2.0      100.0%   98.4%   98.4%    0.6881  -0.0039     0  1024
   4.0       85.2%   63.3%   63.3%  -18.9746  -1.0738    13  1024
   8.0       33.6%   15.6%   15.6%  -19.8983  -7.8469    59  1024
  16.0        5.5%    0.0%    0.0%  -20.4601 -15.1440   107  1024
```

三条自洽性检查：

1. **自由落体**：无地面 ⇒ 2 s 应落 ≈19.6 m，实测 `z_min` −18.97 / −19.90 / −20.46（hold 实际 2.05 s ⇒ 20.6 m）—— 掉落是真的、量级正确；`ovf` 全程只有 `LS_ITERATIONS`，没有接触丢失。
2. 剂量-响应单调：`grasp@reset` 100/100/100/100/85/34/5.5 %，`fell` 0/0/0/0/13/59/107。
3. 与 M1 独立吻合：臂长 ~0.45 m × 臂噪声 0.01 rad ⇒ 1×≈4.5 mm、2×≈9 mm、4×≈18 mm，而 M1 测出的抓握窗口是 1–2 cm ⇒ 边界正好落在 2×（98.4 %）。

## 9.14 第二次 OOM：`max_replay_size` 是**每个 env** 的容量

第一次真跑（`brach_C`，128 envs）在 `buffer_state = jax.jit(replay_buffer.init)(buffer_key)` 处挂掉：

```
XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8652800000 bytes.
```

8,652,800,000 B 不是随机的，正好等于

$$100000 \times 128 \times 169 \times 4 = 8\,652\,800\,000$$

因为 JaxGCRL 的队列是这么分配的（`jaxgcrl/utils/replay_buffer.py:50`）：

```python
self._data_shape = (max_replay_size, num_envs, data_size)
```

即 **`max_replay_size` 是"每个 env 的时间片数"，总占用 = × num_envs**；而 `data_size` = obs 151 + action 14 + reward/discount/truncation/traj_id 4 = **169 floats = 676 B**。

| `max_replay_size` | × 128 envs 的 transitions | 显存 |
|---|---|---|
| **100000（库默认）** | 12.8 M | **8.06 GiB** ← 8 GB 卡必炸 |
| 40000 | 5.12 M | 3.22 GiB（= 上游 512×10000 的总量） |
| **20000（现在的默认）** | 2.56 M | **1.61 GiB** |
| 10000（上游 train.sh 的值） | 1.28 M | 0.81 GiB |
| 6000 | 0.77 M | 0.48 GiB |

修法（`src/train.py`）：
* 默认从库的 `100000` 改成 **`20000`**（1.61 GiB）；
* 新增 `--buffer-gb`（默认 **2.0**）内存闸门：按 `data_size`（用 `flatten_util.ravel_pytree` 现算，不靠硬编码）反推 `cap`，超了就**夹下来并打印警告**，最后把生效值回写 `args.json`；
* 启动时打印 `[buffer] max_replay_size=... x ... envs x 169 floats = ...M transitions, ~X GiB`。

### 9.14.1 8 GB 卡上的显存预算（实测 4060：8188 MiB，空闲时只占 15 MiB）

| 项 | 估算 | 依据 |
|---|---|---|
| replay buffer | 0.81 / **1.61** / 3.22 GiB（10000 / 20000 / 40000 片） | 精确公式 `mrs × num_envs × 169 × 4` |
| Warp `mjx.Data` | 每实例 ~0.3 GiB（128 worlds，`naconmax=16384,njmax=512`，未批量实测 2.76 MiB）；env state + `info["first_pipeline_state"]` + eval env ≈ 2–3 份 | 实测 + 外推；**buffer 不存 pipeline_state**（只有 obs/action/reward/discount/state_extras，已核对 `crl.py:316-332`） |
| 网络 + Adam 状态 | ~0.1 GiB | W=256、`--n-hidden 17`（3 套网络共 3.30 M 参数，+Adam ≈ 38 MiB） |
| XLA 可执行体 / 临时区 | ~0.3–0.5 GiB | — |

⇒ 默认 20000 片时总量约 **4 GiB**，落在 JAX 默认预分配的 75%（≈6.1 GiB）之内，不必动 `XLA_PYTHON_CLIENT_MEM_FRACTION`。
想顶到 40000 片（3.22 GiB）时，建议同时 `XLA_PYTHON_CLIENT_MEM_FRACTION=.90` 把池子放大到 ≈7.3 GiB。

顺带确认：**Warp 的接触预算是便宜的**，之前把 `naconmax` 从 1024 提到 16384 不会造成显存问题——实测 `mjx.Data(impl=warp)` 在 `naconmax=16384,njmax=512` 下只有 **2.76 MiB**（未批量）。

> 若仍 OOM：降 env 数比降 buffer 更划算（同样的显存可以换更多时间片）：
> `--num-envs 64 --batch-size 256 --max-replay-size 40000` → 1.61 GiB、2.56M transitions。

## 9.15 网络结构核对（对照 `docs/CRL设置/模型配置的确定.md`）

`jaxgcrl/agents/crl/networks.py` 里只有两个模块：`Encoder`（critic 用）与 `Actor`。CRL 一共建三套网络：

| 网络 | 输入 | 结构 | 输出 | 实测参数量（当前 = 上游默认） |
|---|---|---|---|---|
| Actor π | `obs = [state(141) \| goal(10)]` = 151 | 主干 `Dense_0..15`(256) + 双头 `Dense_16`=mean、`Dense_17`=log_std；**无 LN**（上游不传 `use_ln`） | 14 + 14 | **1,032,988** |
| φ(s,a) | `[state(141) \| action(14)]` = 155 | `Dense_0..15`(256) → `Dense_16`(256→64)，每层后接 LN | $z_{sa}\in\mathbb R^{64}$ | 1,051,456 |
| ψ(g) | `goal(10)` | `Dense_0..15`(256) → `Dense_16`(256→64)，每层后接 LN | $z_g\in\mathbb R^{64}$ | 1,014,336 |

合计 **3.099 M 参数（12.40 MB）**，带 Adam 的 m/v 约 **35.4 MiB**（用 flax 实际 `init` 数出来的，不是估的；critic 每层 = `Dense_k` 的 kernel+bias + 对应 `LayerNorm_k` 的 scale+bias=512）。
**当前默认 `--n-hidden 17`**（= 论文参考实现的 depth-16）：`Dense_0` 是 stem，之后 `Dense_4 / Dense_8 / Dense_12 / Dense_16` 处各闭合一个 4-unit 残差块，共 **4 次** `h = h + u4`，块内共 16 个 Dense。
若退回 `--n-hidden 16`（JaxGCRL 的默认）则只剩 **3 次**残差，`Dense_13,14,15` 变成没有残差的尾巴。
参数量随之变化（flax 实测）：`D=16` → actor 1,032,988 / sa 1,051,456 / g 1,014,336，合计 3.099 M；**`D=17` → 1,098,780 / 1,117,760 / 1,080,640，合计 3.297 M（13.19 MB，+Adam ≈ 37.7 MiB）**。
没有 Q 头：critic 就是 `logit = bias + scale · raw(φ(s,a), ψ(g))`，`raw = -‖φ-ψ‖²`（预设 C）。

**残差结构：有，而且和文档 §3 的 `h' = h + u4` 完全一致**（这条是数值验证过的，不是读代码猜的）：
写了个小实验，用同一份参数手写了三种前向，和 `Encoder.apply` 比：

```
A code-as-written        max|diff| = 3.278e-07   == reference
B doc h+u4 (stem first)  max|diff| = 3.278e-07   == reference
C wrong (first unit)     max|diff| = 3.482e-01   DIFFERS
```

即：每个 unit 是 `Dense(256) → LayerNorm → Swish`；**每 4 个 unit 做一次 `h = h + u4`**（在 block 最后一次 activation 之后），与论文正文/文档 §3 一致。判别性也验了（故意写错的变体差 0.35，不会被"随便什么都匹配"糊弄过去）。

⚠️ **depth 口径：论文参考实现 vs JaxGCRL 重实现，两者差一个 stem。** 查了论文官方仓库 [`wang-kevin3290/scaling-crl`](https://github.com/wang-kevin3290/scaling-crl) 的 `train.py`（2026-09-18 拉取），网络是这样写的：

```python
def residual_block(x, width, normalize, activation):
    identity = x
    x = Dense(width)(x); x = normalize(x); x = activation(x)   # unit 1
    ... 重复 4 次 ...
    x = x + identity                    # 残差加在第 4 个 unit 的 activation 之后
    return x

# SA_encoder / G_encoder / Actor 一律：
x = Dense(width)(x); x = normalize(x); x = activation(x)      # "Initial layer" ← stem
for i in range(self.network_depth // 4):
    x = residual_block(x, ...)                                # ← 真·残差块
x = Dense(64)(x)                                              # "Final layer"
```

所以**论文的 depth = 4N 是严格成立的**（stem 在循环外、单独一层，不计入 depth），用户记忆无误；`depth=16` = 4 个残差块 × 4 Dense。

而 JaxGCRL 的 `Encoder`（我们实际在跑的代码）把 stem 折进了 `network_depth`，并且把残差加在了"下一个块的第 1 个 unit 之后"：

- `--n-hidden 16`（JaxGCRL 默认）= stem + **3** 个闭合残差块 + 3 层无残差的尾巴；
- **`--n-hidden 17` = 论文参考实现的 depth-16**（stem + 4×4=16 Dense，4 次 `h+u4`）。

数值验证：把 JaxGCRL 的 `Encoder` 与手写的"stem + 4 单元块 `h+u4`"对拍，`n_hidden=17` 时逐位一致（3.3e-7）；把残差加错位置则会差 0.35。**已按决定把默认设为 `--n-hidden 17`**（`src/train.py`），以对齐论文参考实现的拓扑。

> 官方参考实现里还有两点与 JaxGCRL 不同，记录下来以备对照：① 他们的 critic energy 写死为 `-sqrt(Σ(φ-ψ)²)`（= JaxGCRL 的 `norm`），不是平方；② 他们的 `Actor` 也用 `norm_type="layer_norm"`，即**论文的 actor 是有 LN 的**——JaxGCRL 的 `CRL` 忘了把 `use_ln` 传给 `Actor`（见下条，已按"不改上游"的要求回退）。

⚠️ **上游遗漏（记录，不改）**：`Actor` 类定义了 `use_ln` 字段，但 `CRL` 建 actor 时只传了 `use_relu`、**没传 `use_ln`** ⇒ `--use-ln 1` 实际只作用在两个 critic encoder 上，**actor 是纯 `Dense+Swish`**（实测 actor 参数里 `LayerNorm` 叶子 = 0，两个 encoder 各 32 个）。这与论文参考实现不一致，但按"网络保持默认、不擅自改"的要求，**我们已经把之前加的 `actor_use_ln` / `--actor-ln` 全部回退**，现在与上游逐字一致；这条只作为"我们与论文的已知差异"登记在偏差表里。

补丁文件 `patches/jaxgcrl_crl_losses.patch` 现在只含两处**必要的**改动（50 行）：`crl.py` 的 assert→warning + `params` 无条件导出（不修就在无 checkpoint 时 `UnboundLocalError`），以及 `evaluator.py` 允许透传我们的自定义指标。**`losses.py` 与上游逐字节相同**（`git diff --exit-code` 通过）。

### 9.15.1 `logit = bias + scale · raw` —— **已删除（2026-09-18 决定）**

曾经按 `docs/CRL设置/损失函数设置.md` §5/§6 给 `energy_fn` 加过一个仿射 logit（`bias + scale·raw`），用于"D: L2 + balanced binary sigmoid"那一格。**现在这个改动连同那套二分类损失一起删掉了**：

| 曾经改的 | 现状 |
|---|---|
| `losses.py: energy_fn(..., bias, scale)` | **回退**，与上游逐字节相同（`energy_fn(name, x, y)`） |
| `crl.py: logit_bias / logit_scale` 字段 | **删除** |
| `losses.py: binary_nce` 改成 balanced SigLIP | **回退**为上游原版（上游这版没有正负 label，是被那篇分析当作"另一种情况"处理的） |
| `train.py` 预设 `B_dot_binary` / `D_l2_binary` | **删除** |
| `train.py: --actor-ln` / `crl.py: actor_use_ln` | **删除**（网络保持上游默认） |

**当前默认 = 上游原生组合：「L2 能量 + forward InfoNCE」（`--preset C_l2_infonce`，`logsumexp_penalty_coeff=0.1`），`logit` 就是 `energy_fn` 的输出本身，没有任何仿射/温度（`cosine` 分支的上游 CLIP 温度仍在，只在 `energy_fn="cosine"` 时生效）。**

保留的预设只有上游原生支持的组合：`C_l2_infonce`（默认）、`A_dot_infonce`、`upstream_norm`、`S_l2_syminfonce`。
其中 `upstream_norm` = JaxGCRL 出厂的 `energy_fn` 默认值，也等于论文官方仓库里写死的 `-sqrt(Σ(φ-ψ)²)`；如果要把结果与 Scaling-CRL 论文逐项对齐，就换这个（一行 flag）。

## 9.16 「为什么这里会出现 ReLU？」——一个 `args` 笔误 + 静态检查

第一次真跑在 `[net]` 那一行崩了：

```
AttributeError: 'Namespace' object has no attribute 'use_relu'
```

原因：`train.py` 把 `use_relu` **硬编码**成 `False` 传给 `CRL`（论文/上游默认就是 Swish），但我加的那行启动打印里写了 `args.use_relu` —— 而 `--use-relu` 这个 flag **从来没定义过**。**激活函数一直是 Swish，没有 ReLU**；错误只出在打印语句，训练还没开始（env/buffer 已经建好，但 wandb 与训练循环都没进）。

修法（两处，都不改变数值行为）：
1. 把激活变成一个**单一来源**的模块级常量，`CRL(...)` 与打印都引用它：
   ```python
   USE_RELU = False   # Swish everywhere (paper's recipe / upstream default)
   ...
   use_ln=bool(args.use_ln), use_relu=USE_RELU,
   ```
   这样"打印说的"和"实际传的"不可能再分叉。没有加 `--use-relu` flag：论文 ablation 只是警告不要换 ReLU（`docs/CRL设置/模型配置的确定.md` §4 也是这个立场），保持固定即可。
2. 新增 **`src/check_args.py`**：纯 AST 静态检查，把每个脚本里"读到的 `args.X`"与"声明的 `--x-y` / 解析后赋值的 `args.X`"对比，任何读而未声明的名字直接 FAIL。这类笔误只会在大任务跑到一半时暴露，值得固化成工具：

   ```bash
   .venv-warp/bin/python src/check_args.py
   # OK   src/train.py               flags= 33 used= 33 missing=[]
   # OK   src/check_reset.py         flags=  9 used=  9 missing=[]
   # ... 8 个入口脚本全部 OK
   ```

顺带清掉了崩溃时留下的 `runs/brach_C/`（只有 `args.json`，没有 `progress.csv`，说明训练从未开始）。重跑同一条命令即可。

## 9.17 第三次内存事故：Warp 的分配器在 eval 时炸了（训练本身已经跑起来了）

这次**训练真的跑起来了**：第一次 eval 打印

```
[   372992] episode_success=0  episode_dist=1383  episode_max_bar=0  critic_loss=5.796  sps=2231
```

（128 envs 下 2231 env-steps/s，可接受；`dist=1383` 是随机策略掉到 z≈−19 m 后的 10 维 goal 距离，符合预期。）

然后在**第二次** `evaluator.run_evaluation` 崩了，而且不是 JAX 报的错，是 **Warp 自己的分配器**：

```
Warp CUDA error 2: out of memory (wp_alloc_device_async)
  collision_driver.collision -> create_collision_context(d.naconmax)
  collision_core.py:525  collision_pair=wp.empty(naconmax, dtype=wp.vec2i)
RuntimeError: Failed to allocate 131072 bytes on device 'cuda:0'
```

`131072 = 16384 × 8` —— 正好是 **eval env 的 `naconmax`**。

### 根因

1. **`naconmax` 是全局预算，但我们把训练 env 的值原样给了 eval env**：`env_kwargs` 被两个 env 共用，于是 eval env（只有 32 个 world）也拿了 `128 × 128 = 16384`，等于每 world 512 个接触 —— 8 倍的浪费。而 Warp 的 collision context 正是按 `naconmax` 分配的。
2. **Warp 的显存不在 JAX 的池子里**：`XLA_PYTHON_CLIENT_MEM_FRACTION` 默认 0.75，JAX 一上来就把 8 GB 里的 6.1 GiB 占住；Warp 的 `mjx.Data`（含 `naconmax` 尺寸的接触/约束数组，且这些数组是**被 scan 携带的状态的一部分**）与每步的 collision context 只能从剩下的 ~2 GiB 里拿，第一个 eval 用掉了余量，第二个 eval 就不够了。
3. 日志里那句 `only reduced to 4.94GiB ... by rematerialization` 就是 eval 那份 501 步 scan 可执行体的临时内存需求 —— **eval 是最大的单项开销**。

### 修法

| 改动 | 效果 |
|---|---|
| eval env 单独算 `naconmax = per_world × num_eval_envs`（128×16 = **2048**，而不是 16384） | collision context 与 Data 里的全局数组都小 **8×**，直接消掉那个 131072 B 的分配压力 |
| `--num-eval-envs` 默认 32 → **16** | eval 那份 scan 的 world 数减半 |
| 新增 `--xla-mem-fraction`（默认 **0.70**，在 `import jax` 之前设好环境变量） | 从 JAX 池子里让出 ≈400 MiB 给 Warp |
| `[mem]` 启动打印 | 一眼看到当前比例 |

启动时会看到：

```
[eval] goal_bar=4 envs=16 naconmax=2048 (train goal_bar~U[0, 5))
[env] mjx-warp scene=full action=14 obs=151 state_dim=141 n_frames=10 (50 Hz) naconmax=16384 njmax=512 ls_iterations=20
[mem] XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 (MJX-Warp allocates outside this pool -- leave it room on an 8 GB card)
```

### 如果再遇到内存问题，按这个梯子调

| 症状 | 动作 |
|---|---|
| Warp `out of memory`（`wp_alloc_device_async`） | `--xla-mem-fraction 0.65`（再不行 0.60），给 Warp 让位 |
| 还不行 / 想更省 | `--max-replay-size 12000`（1.0 GiB）、`--naconmax-per-world 64`、`--num-eval-envs 8` |
| JAX 报 `RESOURCE_EXHAUSTED` | 反过来把 `--xla-mem-fraction` 调回 0.75 |

> 注意 `--naconmax-per-world 64` 会相应缩小接触预算；改完先用 `src/check_reset.py --naconmax-per-world 64` 确认 `ovf` 里没有 BROADPHASE(4)/NARROWPHASE(8) 位。

## 9.18 每次 eval 之间隔多少步？以及 `eval/episode_*` 到底是**求和**不是平均

**每两行 print 之间 = 238,080 env steps**，算法（`crl.py:202-205`）：

```
env_steps_per_actor_step      = num_envs x unroll_length            = 128 x 62 = 7,936
num_prefill_env_steps         = min_replay_size x num_envs          = 1000 x 128 = 128,000   (名义值)
num_training_steps_per_epoch  = (total_env_steps - 128,000) // (num_evals x 7,936)
                              = 4,761,600 // 158,720 = 30 个 actor step
=> 每个 epoch（= 一次 eval 之间）= 30 x 7,936 = 238,080 env steps
```

对照实测：`611,072 - 372,992 = 238,080` ✓；第一次打印在 `372,992`，因为**真实** prefill 是 `ceil(1000/62)=17` 个 actor step = `134,912` 步（库里公式用的是名义值 128,000），再加 238,080。
外层循环 `for ne in range(config.num_evals)` ⇒ **一共 20 行**，最后一行在 `134,912 + 20 x 238,080 = 4,896,512` 步（比请求的 4,889,600 多 0.14%，因为真实 prefill 比名义值多 6,912 步）。
按实测 sps ≈ 2,100–2,400：每个 eval 间隔 ≈ 100–115 s，整跑 ≈ **35–40 min**。

⚠️ **`eval/episode_*` 是"整条 episode 的求和"，不是平均**（`brax/envs/wrappers/training.py::EvalWrapper.step`：
`episode_metrics = tree_map(lambda a, b: a + b * active_episodes, ...)`）。所以：

- `episode_dist / avg_episode_length` 才是**每步平均距离**；
- **episode 长度不同的两行，`episode_dist` 不能直接比**（掉得越早，累加的步数越少，数字越小，看着像"进步"）；
- `episode_fell`（每步 0/1 的求和，再对 16 个 eval env 取平均）= **掉落 episode 的比例**，可以直接当百分比读 —— 这条更正 §9.10 里"fell 聚合后可能读 0、不可信"的说法：只要 episode 因掉落而结束，那一步在 `active=1` 时就被加进去了，所以 `fell>0` 一定意味着掉了，`fell=0` 一定意味着没掉（在截断长度内）。

本次 run 前三行的正确读法：

| steps | avg_ep_len | dist/步 | 掉落比例 | 换手 | `logits_pos − logits_neg` |
|---|---|---|---|---|---|
| 372,992 | 501（满） | 2.778 | 0 % | 1.0 | 2.61 |
| 611,072 | 501（满） | 2.924 | 0 % | 2.0 | 7.49 |
| 849,152 | **233** | 2.946 | **69 %** | 6.3 | 10.83 |

- 起点的 `dist/步 ≈ 2.78` 正好等于 `‖goal(B4) − goal(B0)‖ = 2.7713` —— 即未训练策略**原地挂着**（没有掉、也没前进），这比"乱摔"更健康；
- 第 3 行开始 `avg_ep_len` 从 501 掉到 233、`fell=69 %`：策略学会了动，但还没学会**不掉**（CRL 早期的正常阶段，第一个要过的关口就是"别掉"）；
- 唯一稳步变好的量是 `logits_pos − logits_neg`（2.6 → 7.5 → 10.8）：critic 在学会区分正/负样本，这是"栈没问题"的最直接证据；
- 判断"有没有前进"请看 `eval/episode_max_bar`（现在还是 0）与 `eval/episode_forward / avg_episode_length`（现在 ≈ −0.03 m/步，几乎没动）。

## 9.19 实测吞吐与单次 run 的墙钟（4060 Laptop 8 GB）

从当前这次 run 的 `progress.csv` 反推（`training/walltime` 与 `eval/walltime` 都是**累计值**，相减得每窗用时）：

| 窗口 | steps | 训练 | eval | 合计 | 端到端吞吐 |
|---|---|---|---|---|---|
| 1 | 372,992 | 114.0 s（含首次编译） | 16.8 s（含评测图首次编译） | 130.8 s | — |
| 2 | 611,072 | 97.8 s | 7.19 s | **105.0 s** | 2,268 env-steps/s |
| 3 | 849,152 | 99.2 s | 8.13 s | **107.3 s** | 2,218 env-steps/s |
| 4 | 1,087,040 | 99.3 s | 8.71 s | **108.0 s** | 2,204 env-steps/s |

每窗固定 `238,080` env steps（§9.18）。稳态 **≈ 108 s/窗**（训练 99 s + 评测 9 s）。

**本次 run（4.896 M env steps，20 窗）的墙钟预算**

| 项 | 时间 |
|---|---|
| 编译 + prefill（未计入上面的累计值，按启动日志估） | ~150 s |
| 第 1 窗（含训练图与评测图首次编译） | 114 + 17 s |
| 第 2–20 窗（19 × 108 s） | ~2,052 s（训练）+ ~171 s（评测） |
| **合计** | **≈ 2,300 s ≈ 39 min** |

**吞吐随规模的线性外推（按 2,200 env-steps/s，含 eval）**

| 总步数 | 墙钟 |
|---|---|
| 4.9 M（本次基线） | ≈ 37 min |
| 10 M | ≈ 1.3 h |
| 20 M | ≈ 2.5 h |
| 50 M | ≈ 6.3 h |
| 3 seeds × 10 M | ≈ 3.8 h |

注意：① 首次编译/评测编译是固定开销（~2.5 min），长跑会被摊薄；② 笔记本 4060 会降频，长时间跑的实际 sps 可能比现在低 5–15%；③ 换 `--num-envs` / `--batch-size` / 网络深度都会改变吞吐，这张表只对当前配置（128 envs、batch 512、W=256、D=17）成立。

## 9.20 决定：3 小时那次 run **加 `--train-goal-bar-min 1`**

依据（不是拍脑袋，是上游的设计约定）：

| 上游环境 | 指令目标怎么采 |
|---|---|
| `jaxgcrl/envs/humanoid.py` | `_random_target`: `dist ~ U[min_goal_dist, max_goal_dist]`，默认 **`min_goal_dist=1.0`**, max=5.0 —— 目标**永远不可能就是起点** |
| `jaxgcrl/envs/ant_maze.py` | goal 从 `possible_goals` 里抽，**start 从 `possible_starts` 里抽**，两套集合独立 |

也就是说：**JaxGCRL/论文的惯例是"指令目标永远是一个尚未达成的目标"**。我们原来 `goal_bar_min=0` 等于每 5 个 episode 就有 1 个被指令"去你已经在的地方"——这是采样 5 个标准悬挂帧时无意带入的偏离，3 小时的 run 里 20% 的数据花在"已经达成的指令"上。

为什么这个改动**几乎是免费的**：

- 评估侧不受影响：`eval_env` 一直固定 `goal_bar=4`，两种配置在评估时看到的是同一个目标；
- 训练侧真正产生梯度的是**未来帧重标记**出来的 goal（`goal = future_obs[:, goal_indices]`），与指令目标无关；`bin=1` 只改变了**行为策略被指令去做什么**，即数据分布偏向"试着移动"；
- "保持不动"这个课程**不会因此消失**：一条原地不动的轨迹，它的未来帧本身就近似当前状态，重标记后依然是"stay"类正样本，critic 照样拿得到。

代价（要如实记）：

- 与刚跑的 5 M 步那次不再逐项可比 → 换个 `--exp-name`（`brach_C_gmin1`），把 5 M 那次定位为"栈自检"，不作为科学基线；
- 早期"学会别掉"的阶段可能略长（策略再也没有被指令"原地待着"）→ 前 2 个窗口盯 `eval/episode_fell` 与 `avg_episode_length`。上次是 ~1.3 M 步恢复的；
- 这正是执行计划 M3 里列的第一个杠杆，所以这次 run 顺带完成了一个消融记录。

**3 小时对应的命令（`steps=24e6`，实测 ≈2,200 env-steps/s 端到端）**

```bash
.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --num-evals 40 --steps 24000000 --train-goal-bar-min 1 \
  --impl warp --scene full --wandb --exp-name brach_C_gmin1
```

（`num_evals=40`：每窗 595,200 步 ≈ 4.5 min，曲线点从 20 个变 40 个，评估额外开销只多 ~3 min，用来抓"第一次跨杆"这种稀有事件更划算。`num_evals=20` 则每窗 1,190,400 步 ≈ 9 min，总时长一样。）

## 9.21 3 小时 run（24 M 步）的结论：**策略收敛到"挂在 B0 上抖手"，一次都没跨杆**

`runs/brach_C_gmin1/`（40 个 eval，`train_goal_bar_min=1`，`n_hidden=17`，seed 0，训练 175.7 min + 评测 6.9 min = 3.0 h）：

| 指标 | 全程范围 | 读法 |
|---|---|---|
| `eval/episode_max_bar` | **恒为 0（40/40）** | 从未抓过 B0 以外的杆 |
| `dist / avg_episode_length` | 2.752 – 3.139（基线：B0 挂姿到 B4 目标 = **2.7713**） | 没有前进，只有一次早期掉落期的抖动 |
| `forward / len`（躯干 x 偏移） | −0.073 … −0.014 m（杆距 **0.4 m**） | 始终停在 B0 附近 ±7 cm |
| `eval/episode_fell` | 20 M 步里 0%，两个窗口塌到 75%/100% 后恢复 | 稳得住，不是乱摔 |
| `bar_L` / `bar_R`（几何代理） | L≈0（常抓 B0）；R 在 0 与 −1 间来回 | 右手反复松/抓**同一根杆** |
| `episode_hand_switches` | 1 – 12 / episode | 接触切换**行为存在**，但都在 B0 上 |
| `logits_pos − logits_neg` | 6.3 → **11.6** | critic 一直在学，最后收敛 |
| `categorical_accuracy` | 0.19 → 0.53（随机 0.2%） | 同上 |
| `entropy` | 锁定在目标 −7.00 | α 调节正常 |
| `critic_loss` | 4.40 → 1.86 | 同上 |

**判定：栈完全正常，任务的第一步（离杆）从未被发现 —— 属于探索失败/局部最优，不是 bug。**

机制（为什么自课程救不了它）：CRL 的 goal 是从**已收集轨迹的未来帧**重标记出来的。策略从未离开 B0 流形 ⇒ buffer 里根本没有 B1 及以上的状态 ⇒ 重标记出的 goal 永远在 B0 附近 ⇒ actor 的均值收敛到"原地保持"。被指令的 goal（B1..B4）**不进入 actor/critic 的梯度**（`update_actor_and_alpha` 用 `future_state[:, goal_indices]`），所以改变指令分布（`gmin=1`）没能改变结果。探索噪声确实在抖手（1–12 次/episode），但"释放→弹射伸手→抓住 0.4 m 外的下一根杆"这种需要协同的低概率事件，靠无结构噪声撞不出来。

### 9.21.1 附带发现：**这次 run 没有存下参数，无法回放视频**

`args.json` 里 `checkpoint_dir: null`，`config.checkpoint_logdir` 默认也是 None ⇒ 24 M 步的策略事后不可恢复。已补两个洞：

1. **`src/train.py`**：`--checkpoint-dir DIR` + 新增 `--save-every N`（默认 10 个 eval）→ 每个 checkpoint 写 `actor_<steps>.pkl` 与 `actor_latest.pkl`，内容是 **actor 参数 + 重建网络/env 所需的 config**；结束时仍写 JaxGCRL 格式的 `final`，并额外写一份 actor-only。
2. **`src/render_policy.py`（新工具）**：读上述任一格式（也兼容 JaxGCRL 的 `(alpha, actor, critic)` 元组），按评估口径做**确定性** rollout（`a = tanh(mean(obs))`），再把 mujoco-warp 产生的 `qpos` 用**原生 MuJoCo**回放渲染（所以视频就是训练用的那套物理）。输出 `runs/render/<name>/`：`frames/step_*.png`（无依赖）+ `<name>.mp4`（装了 imageio）或 `.gif`（pillow 兜底）+ `trace.csv`（逐步 qpos / 抓握代理 / 杆号 / 10 维与 position-only 两种 goal 距离）+ 打印总结（到达的杆号、掉落步、换手次数、质心位移）。
   `.venv-warp` 里没有 imageio，要 mp4 就 `uv pip install imageio imageio-ffmpeg`；否则 PNG + ffmpeg 一行命令（脚本会打印）。

```bash
# 下一次 run 存参数 → 之后随时回放
... --checkpoint-dir runs/ckpt_<exp> --save-every 10
.venv-warp/bin/python src/render_policy.py --ckpt runs/ckpt_<exp>/actor_latest.pkl --episodes 4
```

## 9.22 一个被量化出来的结构性障碍：**goal 里的接触项惩罚"松手"**

⚠️ **先澄清两套完全不同的"距离"**，别混：

| | 定义 | 用在哪 | 需要编码器？ |
|---|---|---|---|
| **任务度量**（下表就是它） | $\lVert\text{achieved\_goal}(s)-g\rVert_2$，10 维**原始** goal 空间 | `success`/`success_easy`/`dwell_success`（阈值 0.35）、评估指标、指令目标的构造 | **不需要** |
| **critic 的价值** | $f=\texttt{bias}+\texttt{scale}\cdot\text{raw}(\phi(s,a),\psi(g))$，64 维**学出来的**嵌入空间 | actor 的梯度、`logits`/`critic_loss`（论文的 Q 可视化也是它） | **需要**（φ、ψ 两个 encoder） |

下表算的是**任务度量**：`goal_set` 由关键帧正运动学直接给出（`brachiation.py:196-199`，无网络），中间态是我**手工构造**的 10 维向量（假设躯干不动、手前伸 x 米、该手 c 置 0），不是仿真状态。它说明的是"**我们定义的 goal 空间与成功判据**如何看待松手"，**不是**"critic 学到的度量"——critic 完全可以在 64 维里学出另一套几何（它的目标就是把同一轨迹里可达的对拉近）。

用真实 `goal_set` 算"从 B0 出发、以 B1 悬挂为目标"时各中间态的距离（`src/goal_geometry.py` 的同一套几何）：

| 中间状态 | 10 维 goal 距离 | position-only(8 维) |
|---|---|---|
| 挂在 B0 | 0.693 | 0.693 |
| 松开左手、手仍在 B0 | **1.217** | 0.693 |
| 松开左手、前伸 0.2 m | **1.166** | 0.600 |
| 松开左手、前伸 0.4 m（够到 B1 处） | **1.149** | 0.566 |
| 已挂在 B1（目标） | 0.000 | 0.000 |

**在 10 维任务度量下，"松手"这一步永远让你离目标更远**（`(Δc)²` 贡献 +1，远超位置项的收益）；只有 position-only 的 8 维空间里，"前伸"是一条单调下降的阶梯。影响学习的具体通道有三条：① 评估/成功判据用的就是这个度量；② 重标记出的 goal 是同一 10 维向量的切片，其 c 分量取 1 意味着"抓着"才算到达；③ 指令目标是"双手抓在 Bk"的悬挂态，没有任何东西告诉策略"先松一只手"。

但要说清楚：**这不构成"critic 学不会"的证明**——critic 的度量是学出来的，原则上可以把"松手前伸"映射成一条通向目标的路。所以这是一个**假设**，`position-only(8)` 消融（任务变量文档 §3.5 / Q2 早已预留）就是检验它的实验：把任务度量改成对"前伸"单调，再跑 1e7 步看 `max_bar` 是否出现非零。要直接测"学到的度量"则需要 critic 参数（`final` 元组里有；中途快照目前只存 actor），见 §9.23。

### 接下来的顺序（每次只动一个变量）

| # | 改动 | 成本 | 为什么 |
|---|---|---|---|
| 1 | **goal 换成 position-only（8 维，去掉 `c_L,c_R`）** | ~20 行（`goal_indices` + `goal_set`） | 上表：给"前伸"一条单调下降的梯度阶梯；抓握信息仍在 *state* 里，策略看得见 |
| 2 | **固定指令目标 B1**（`--train-goal-bar 1`，需加 ~3 行 flag） | 很小 | 执行计划 M3 原本就是"单目标 B1"；把行为集中到唯一的下一个动作，而不是摊在 B1..B4 |
| 3 | **易化几何 d=0.30** | 中（重建杆位 + 关键帧重解） | 伸手距离 −25%，M3 的既定兜底 |
| 4 | **主动摆腿 A4**（腿从定值改 1–2 维动作） | 中 | 现在腿是死的，没有泵能手段；弹射伸手在物理上需要摆腿给动量 |
| 5 | **CRL vs SAC+HER 对照** | 大 | 判定"是 CRL 不行"还是"这个任务对任何 GCRL 都难"——计划里的决策点 |

建议：先做 **#1**，跑 1e7 步（≈1.3 h）看 `max_bar` 是否出现非零；出现就延长到 3e7，不出现再叠加 #2，然后 #3/#4。

## 9.23 待办：把"学到的度量"也测出来（需要 critic 参数）

上面 §9.22 的表是**任务度量**。要回答"critic 到底怎么看待松手"，得直接在 64 维嵌入空间里量：

$$f(s,a,g)=\text{raw}(\phi(s,a),\psi(g)),\qquad g=g(B_1)$$

并比较两条路径上的值：`挂 B0 → 松手 → 前伸 → 抓 B1`。如果学到的 $f$ 在"松手"那一步**下降**，就说明 critic 也把松手当成退步（那就该改 goal 定义）；如果它上升或持平，则问题纯在探索，与 goal 定义无关。

所需材料与现状：
* `final`（JaxGCRL 元组，含 `sa_encoder`/`g_encoder`）—— `--checkpoint-dir` 时已经会存 ✓；
* **中途快照目前只存 actor**（`progress_fn` 只拿得到 actor 参数，上游签名如此），所以"随训练演化的价值度量"暂时只能取最后一个点。若要每个快照都带 critic，需要给 `progress_fn` 多传一份参数（改 `crl.py` 一行）——按"少改上游"的原则先不做，等真要用时再加；
* 计算入口：给 `src/render_policy.py` 加 `--critic <final>`，沿 rollout 逐步算 $f$、以及 §9.22 那几条手工构造路径的 $f$，一起写进 `trace.csv`。

## 9.24 第 4 次 run：position-only goal + 固定指令目标 B1（4 h）

配置（相对上次只动两个目标相关的变量，其余全部保持）：

| 项 | 值 | 说明 |
|---|---|---|
| `--goal-position-only 1` | goal = `[x_torso, z_torso, p_L(3), p_R(3)]`（**8 维**，obs 149） | §9.22：去掉 `c_L,c_R`，让"前伸"在任务度量里单调下降 |
| `--train-goal-bar 1` | 训练指令目标**固定 = B1** | 执行计划 M3 的原意（单目标 B1）；把行为集中到唯一的下一个动作 |
| `--eval-goal-bar 1` | 评估目标也 = B1 | `episode_success`/`dwell_success` 这次是"有没有到 B1 悬挂"；跨 run 的可比量仍是 `episode_max_bar` |
| `--steps 33460000 --num-evals 40` | 实际 **33,466,112** 步，窗口 833,280 步 | 按实测 2,405 env-steps/s（训练）+ 9 s/窗口（评测）+ 150 s 启动 ≈ **3.99 h** |
| `--checkpoint-dir runs/ckpt_pos8_b1 --save-every 10` | 5 个 actor 快照（含 final） | 上次就是因为没这个而无法回放视频 |
| 其余 | 128 envs / batch 512 / γ=0.995 / 501 步 / W=256 **D=17** / L2+InfoNCE / buffer 20000 / xla 0.70 / 16 eval envs | 与上次逐项相同 |

新增的两个开关（默认值保持旧行为，不影响已跑过的 run 的复现）：
* `Brachiation(..., goal_position_only=)` / `default_layout(njoints, position_only=)`：只改 goal 的取子集方式；`_achieved_goal` 先构造完整 `[x,z,p(6),c(2)]` 再取子集，所以 8 维与 10 维共用同一条构造路径，永远不会不一致（实测两者到 B1 的距离都是 **0.6928**，因此 `goal_reach_thresh=0.35` 的标定不变）。
* `train.py --train-goal-bar`（≥0 时固定，覆盖 `--train-goal-bar-min`）、`--goal-position-only`；两者都写进 checkpoint 的 `config`，`render_policy.py` 会据此重建网络输入维度与 env，避免形状不匹配。

## 9.25 视频/轨迹揭示的行为：**"松开一只手，但够不到下一根杆"**（run #4 回放）

用户看视频的观察被 `trace.csv` 完全证实，而且比肉眼更清楚：

| 现象 | 数据（4 个 episode，每集 501 步，确定性策略） |
|---|---|
| 全部掉落 | 掉落步 126 / 311 / 187 / 173（即 2.5–6.2 s） |
| 两手都读"没抓"却**没掉** | `c_L<0.5` 占 352–476/501 步、`c_R` 类似，但躯干 z 长期停在 **0.68–0.79**（悬挂高度）→ 靠**手腕/前臂勾住杆**支撑，几何抓握代理看不见这种支撑（"hook 模式"） |
| 向前只走了 1/4 | 躯干 x 最大 **+0.12 m**（起点 +0.022 → B1 需要 **+0.422**，即 0.40 m） |
| 目标距离几乎没动 | 全片最好 `dist_pos` = **0.612–0.630**，起点 0.693 → 只走了约 10% |
| 训练曲线印证 | 40 个 eval 里 `max_bar` 恒 0；`d/st` 中段最好 0.59–0.62，**最后 1/3 崩掉**（`fell` 81–100%、`len` 74–232） |

### 病因

position-only（8 维）确实第一次给出了前进方向上的梯度（0.693 → 0.59，前 3 次 run 是**零**变化），但它把"抓着"整个从目标里去掉了，于是：

1. **张手是免费的** → 策略漂移到"手指张开、腕部勾住杆"这种度量看不见的支撑模式（且最终不稳定，会掉）；
2. **丢掉最后一点抓握也不受罚** → 最后 1/3 的崩塌；
3. 始终没有学会**摆动**（躯干 x 始终在 ±0.09 m 内，没有钟摆）→ 静态悬挂下 0.4 m 的弹射伸手在几何上不可能（臂长 0.35 + 手 ≈ 0.45 m，垂直悬挂时手几乎不能水平移动）。

### 修法：`--goal-variant support`（9 维；默认仍是 `full`，不影响旧 run）

$$\text{goal}=[x_{torso},\,z_{torso},\,p_L(3),\,p_R(3),\,\max(c_L,c_R)]$$

* **松开一只手仍然免费**（另一只手 `c=1`）⇒ 保留 position-only 得到的"前伸单调下降"性质；
* **丢掉最后一点支撑要罚 +1** ⇒ 直接判死 hook 模式与"张手掉下去"；
* `max(c_L,c_R)` 作为 `c_support` 只在 support 变体里额外占用**一个 state 维度**（142），所以 `full`（141/151）与 `position`（141/149）的 obs 尺寸**与旧 run 完全一致**——旧 checkpoint 仍可直接回放（已验证 `--ckpt runs/ckpt_pos8_b1/actor_latest.pkl` 解析为 position、obs 149 与参数匹配）。

三种变体实测（`src/envs/brachiation.py::default_layout`）：

| variant | state_dim | goal_size | obs | 到 B1 的初始距离 |
|---|---|---|---|---|
| `full` | 141 | 10 | 151 | 0.6928 |
| `support` | 142 | 9 | 151 | 0.6928 |
| `position` | 141 | 8 | 149 | 0.6928 |

三者到 B1 的几何距离**都一样**，所以 `goal_reach_thresh=0.35` 与 `episode_max_bar` 仍然跨变体可比。

### 渲染的相机问题（用户看到"杆子在晃"）

杆子是**焊死在世界上的**：`bar0..bar4` 的 `body_jntnum=0`、`parent=world`，`nq=50=7(free joint)+43(机器人关节)` ⇒ 杆子没有任何自由度，物理上不可能动。视频里动的是**相机**：`render_policy.py` 原来默认 `--follow 1`，每帧把 `cam.lookat[0]` 设成躯干 x，于是静止的杆子在画面里滑动。已改成默认固定相机（`--follow 0 --lookat-x 0.8 --distance 3.4`，一屏容纳 5 根杆）；`--follow 1` 仍可手动开启。

### run #5 建议

```bash
.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --goal-variant support --train-goal-bar 1 --eval-goal-bar 1 \
  --num-evals 40 --steps 33460000 \
  --checkpoint-dir runs/ckpt_support_b1 --save-every 10 \
  --impl warp --scene full --wandb --exp-name brach_support_b1
```

判读：① `fell` 应显著低于 run #4（hook 模式被罚）；② `d/st` 若能压到 0.5 以下说明在持续前伸；③ `max_bar > 0` 即 M3 达标。
若 run #5 稳定悬挂但仍不摆动/不前伸，下一个杠杆是 **A4：把 2 个髋关节加进动作空间（14→16 维）**——静态悬挂下必须靠摆腿泵能，而现在是 12 个死关节、34 kg 的刚性摆。


## 9.26 M3.0 物理可达性检查（`src/m3_swing.py`，已完成）

依据 `docs/M2_M3诊断.md` 的第 1 条要求：**先不动 RL，用脚本化周期控制确认"一只手抓 B0、自由手能不能够到 B1"**。
脚本跑在**训练用的同一个资产**上（`scene_bars.xml`，full mesh，$d=0.40$，正手），原生 MuJoCo、无需 GPU/JAX。

实现要点：
* 抓握中心沿用 env 的几何定义（腕部位姿 + 关键帧标定的 `seat_local`），所以数字与 env 的 `_grasp` 直接可比；
* **物理接触**用 `build_scene.contact_report`（真实 MuJoCo 接触，不是几何代理）；
* 三种检查：`--ik`（静态运动学扫描：只动右手臂，身体冻结、左手精确在 B0）、`--refine N`（在 IK 最优附近做**带动力学的**随机搜索 = "dynamic IK"）、默认的周期驱动扫描（`free_arm/free_full/waist/elbow/shoulder/all/reach_ik`，可扫 $f$、幅值、相位、偏置与触发式抓握）；
* `--spacing` 可在内存里改动杆距重建场景；**必须配 `--mjx-compat 0`** 才和训练资产同源（`mjx_compat=True` 的 capsule 版本关键帧不同：`cage_offset` 0.030 vs 0.015、`hip_pitch` −0.822 vs −0.654、`bar_x` 0.0706 vs 0.0556，见 `scene_bars*.json`）。已验证 `--spacing 0.40 --mjx-compat 0` 精确复现 `scene_bars.xml`（`bar_x`、IK 最优 pose 与距离都一致）。

**静态运动学结果**（左手固定抓 B0，只动右手臂）：

| 杆距 | 最优姿态（sp / sr / el） | $\lVert p_R-p_{B_1}\rVert$ | 判据 |
|---|---|---|---|
| **0.40（当前训练资产）** | −2.300 / −0.600 / 2.125 | **0.0310 m** | < 0.05 抓握阈值 ✓ |
| 0.35 | −2.300 / −0.800 / 1.875 | 0.0007 m | 几乎精确 ✓ |
| 0.30 | −2.600 / 0.000 / 2.500 | 0.0023 m | 几乎精确 ✓ |

**动态结果**（PD 伺服 + 真实接触 + 触发闭合，30–40 个手工设计姿态）：

| 杆距 | 最优 $\lVert p_R-p_{B_1}\rVert$ | 是否与 B1 发生物理接触 |
|---|---|---|
| 0.40 | 0.0645 m | 否 |
| 0.35 | 0.0444 m | 否 |

**结论**：几何、力矩与控制器都**不是**瓶颈——即使保持 0.40 杆距，也存在"左手仍抓 B0 且右手抓握中心落在 B1 上"的静态姿态；动态下手工姿态只差 ~1.5 cm 未接触，差的那点是"身体在自由臂反作用力矩下转动 + PD 跟踪误差"，也就是**一次协调的动态摆荡/伸手时序**。这正是策略要学的东西，与诊断文档的判断一致：**卡在数据覆盖/探索，不在 critic 容量、也不在 L2+InfoNCE**。

M3 的后续（覆盖率指标 → goal 语义 → 探索时间结构 → 近目标 reset 诊断）已按诊断文档改写进 `docs/执行计划.md` 的 M3 节（M3.0–M3.5）。


## 9.27 M3.0b：**动作窗太小，前四次 run 根本无法表达"够杆"**（重大发现）

M3.0 的静态 IK 指出：左手仍抓 B0 时，要抓 B1 需要肩 pitch `+0.459`、肩 roll `+0.885`、肘 `+1.611` rad 的关节目标增量。
查动作参数化（`src/envs/brachiation.py`）：

```python
q_des = key_ctrl + ARM_WAIST_SCALE * a        # a in [-1, 1]
ARM_WAIST_SCALE = (0.20 x10 arms, 0.25 waist_yaw, 0.15 waist_pitch)   # 旧表 "m1"
```

即**臂关节只有一个 ±0.20 rad（±11.5°）的窗口**，腰 yaw ±14.3°、腰 pitch ±8.6°。而 IK 姿态需要的动作值是
`(2.29, 4.43, 8.05)` —— 全部远超 ±1，clip 之后只剩 ±0.2 rad 的抖动。**策略在结构上不可能命令出够杆姿态。**

实测（`src/m3_swing.py`；左手抓握全程 100% 保持，接触用真实 MuJoCo 接触判定）：

| 杆距 | 动作窗 | 需要的 a（pitch/roll/elbow） | 动态最小 $\lVert p_R-p_{B_1}\rVert$ | 与 B1 接触 |
|---|---|---|---|---|
| 0.40 | `m1` | (2.29, 4.43, 8.05) → clip 到 ±1 | **0.2913 m** | 否 |
| 0.35 | `m1` | (2.29, 3.43, 6.80) → clip | 0.2424 m | 否 |
| 0.40 | **`reach`** | (0.66, 0.74, 0.81) → 装得下 | **0.0930 m** | 否 |
| 0.35 | `reach` | (0.66, 0.57, 0.68) | 0.0723 m | 否 |

另外用随机搜索量化"这个动作窗到底能表达多近"（静态，身体冻结、左手精确在 B0）：
`window=m1` → 最好 **0.123 m**；`window=reach` → 最好 **0.005 m**（受 `|p_L−B0|<0.02` 约束的版本待补；不加约束时搜索会偷动左臂/腰，读数无意义）。

**这解释了此前所有观察**：`max_bar` 四次长跑全 0、`hand_switches` 能涨到 6–14（开合手指在动作窗内）、躯干 x 摆幅只有 0.12–0.15 m（≈ ±0.2 rad 的手臂动作量）、"松手但不前伸"——**都是动作窗的后果，不是 critic/探索/goal 的问题**。`brach_support_b1`（33.5 M 步，support goal）同样 `max_bar=0`，因此 goal 语义的改动在旧窗下也不可能有效。

修法（已实现，默认生效）：
* `src/envs/brachiation.py` 增 `ACTION_WINDOWS = {"m1": 旧表, "reach": 新表}`，`Brachiation(action_window=...)`；
* `reach` 表（rad/单位动作）：臂 `0.70 / 1.20 / 0.80 / 2.00 / 0.70`，腰 `yaw 1.00 / pitch 0.45`（腰 pitch 覆盖自身 ±0.52 的 87%）；
* `train.py --action-window {m1,reach}`（**默认 reach**）、`check_reset.py` 同；`render_policy.py` 从 checkpoint 的 config 读取；
* checkpoint 的 `config` 里记录 `action_window`，旧 checkpoint（`m1`）仍能按原窗回放。

**下一步顺序（相对诊断文档的调整）**：先把动作窗换成 `reach` 单独跑一次（其余与 `brach_support_b1` 相同），看 `max_bar` 是否终于非零；再做 M3.1 覆盖率指标 / M3.2 goal 语义 / M3.3 探索时间结构。

### 9.27.1 run #6（M3.0b）的启动命令与注意事项

run #5（`brach_support_b1`）的 `args.json` 里没有 `action_window` 字段——它跑在旧的 `m1` 窗（臂 ±0.2 rad）上，因此**所有旧 run 都受同一个动作窗限制**。run #6 只改这一个变量：

```bash
# A) 先 1.5 h 快看（推荐先跑这条）
.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant support --train-goal-bar 1 --eval-goal-bar 1 \
  --num-evals 20 --steps 12200000 \
  --checkpoint-dir runs/ckpt_reach_b1 --save-every 5 \
  --impl warp --scene full --wandb --exp-name brach_reach_b1_short

# B) 通过后跑满 4 h（其余完全相同，只把窗口数与步数放大）
.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant support --train-goal-bar 1 --eval-goal-bar 1 \
  --num-evals 40 --steps 33460000 \
  --checkpoint-dir runs/ckpt_reach_b1 --save-every 10 \
  --impl warp --scene full --wandb --exp-name brach_reach_b1
```

步数换算（实测 2,405 env-steps/s 训练 + 9 s/窗口 + 150 s 启动）：A = 12,197,632 步 ≈ **1.48 h**（每窗 603,136 步）；B = 33,466,112 步 ≈ **3.99 h**（每窗 833,280 步）。

⚠️ **这一改同时放大了物理探索噪声（必须盯前两个窗口）**：`q_des = key_ctrl + scale·a`，而 CRL 的 target entropy 写死为 `−0.5·action_size`（与 scale 无关），所以同样 σ_a 下**关节空间的噪声 RMS 放大 5.62 倍**（`sqrt(Σscale_reach²/Σscale_m1²)`）。为把物理噪声拉回原水平，新增了 `--entropy-param`（默认 0.5 = 上游）并通过 `CRL.entropy_param` 生效：

* 若 A 的前两窗 `eval/episode_fell` 直接飙到 100%、`avg_episode_length` 掉到几十步 → 就是噪声把抓握打散了，改用 **`--entropy-param 2.23`**（目标熵 −7 → −31.2，正好抵消 5.62 倍）重跑；
* 若前两窗仍然 `fell≈0`、`len≈501` → 说明噪声可接受，直接跑 B。

## 9.28 run #6（`brach_reach_b1_short`，1.5 h，reach 窗）的结果与解读

配置：`--action-window reach --goal-variant support --train-goal-bar 1 --eval-goal-bar 1 --num-evals 20 --steps 12200000`（其余同 run #5；**只改了动作窗**）。20 个 eval，落地 12,197,632 步。

**好消息：`eval/episode_max_bar` 第一次出现非零**（第 7,975,680 步起：0.44 → 之后 2.9–5.4），即**右手抓握中心第一次进入 B1 的 5 cm 判据**。前四次长跑共 ~9,000 万步里这个量恒为 0 —— 动作窗确实是此前的硬门槛，M3.0b 的判断被证实。

**坏消息：策略随后陷入"乱抓 + 掉"**：

| steps | len | d/步 | fell | max_bar(sum) | 每步换手率 | pos−neg |
|---|---|---|---|---|---|---|
| 2,547,456 | **496** | 0.796 | 19% | 0 | 0.016 | 12.5 |
| 3,753,728 | 134 | 1.04 | 100% | 0 | 0.037 | 10.8 |
| 7,975,680 | 70 | 1.41 | 100% | **0.44** | 0.111 | 8.9 |
| 12,197,632 | 60 | 1.33 | 100% | 4.4 | **0.175** | 6.9 |

对照 run #5（m1 窗）末三窗：`len=472, fell=25%, 换手率 0.011, max_bar=0, pos−neg=13.3`。
**关键对照：两者的 `training/entropy` 都是 −7.00（目标 −7）**，而 reach 窗的关节空间噪声 RMS 是 m1 的 **5.62 倍**（√(Σscale²) 比）——于是换手率涨了 **14 倍**、抓握被打散、episode 从 ~500 步掉到 ~60 步。这正是 §9.27.1 预警的副作用，现在有了实测证据。

**判读**：动作窗修好了（可达性 ✓），但**探索噪声幅度**在放大后的窗口里过大，把"保住抓握"这个前提毁掉了；critic 也随之退化（`pos−neg` 13.3→6.9，因为 buffer 被短命掉落轨迹填满）。2.55 M 步那一窗（`len=496`、`fell=19%`、`d/步=0.796`）说明"稳住+轻微前移"是可达的状态，只是没保持住。

### 9.28.1 下一步：先降噪声幅度，再给噪声时间相关性

新增两个开关（都属于 M3.3 的范畴，已实现）：
* `--entropy-param`：目标熵 = −param × 14（上游写死 0.5 ⇒ −7）。要抵消 5.62 倍噪声需 `2.23`（目标 −31.2）；折中值 `1.2`(≈2× 降噪) / `1.6`(≈3× 降噪)。
* `--expl-hold K`：同一份探索扰动保持 K 个 control step（默认 1 = 上游逐步独立）。50 Hz 逐步独立噪声在 20 ms 内自相抵消，而摆荡需要 0.1–0.5 s 的连续同向作用；K=10 即 0.2 s。

建议阶梯（每次 1.5 h，同样命令只改这两个开关）：
1. **A2 = `--entropy-param 1.6`**（噪声降到 1/3，i.i.d.）→ 看 `fell`/`len` 是否回到 ~500，`max_bar` 是否保住非零；
2. 若 A2 稳但 `max_bar` 归零 → **A3 = A2 + `--expl-hold 10`**（幅度不变、时间相关）→ 这是"既稳又能探索"的组合；
3. 若 A2 仍 `fell=100%` → 直接用 `--entropy-param 2.23`（与 m1 的物理噪声等强度）。

### 9.28.2 覆盖率/极值指标已加入 env（M3.1）

`src/envs/brachiation.py` 新增 `_coverage()`（纯记录，**不改物理、不改 obs/state**，所以旧 checkpoint 仍可用），利用"评估器对每步指标**求和**"这一事实把求和量设计成可直接解释：

| 指标 | episode 求和 = | 均值含义 |
|---|---|---|
| `cov_cross_b1_{050,030,020,010}` | 1（若能进入该半径）/ 0 | **C(r) = P(min 手-杆距离 < r)**，即诊断文档要的覆盖率 |
| `cov_min_d_b1_improve` | `D0(2.0) − min_t d_B1` | `2.0 − 值` = **episode 内自由手到 B1 的最近距离** |
| `cov_swing_improve` | episode 内（有抓握时）躯干最大水平摆幅 | 真实摆幅极值，而非被长度稀释的均值 |
| `cov_fwd_improve` | episode 内最远手的前伸量（相对 bar0） | 是否真的往前伸过 |
| `cov_speed_improve` | episode 内最大 root 线速度 | 摆荡强度 |
| `cov_hand_on_steps` | 至少一只手抓着的步数 | 支撑时长 |

数值已校验（关键帧：`d_B1=0.4` ⇒ `C(0.5)=1, C(0.2)=0`，`min_improve=1.6=2.0−0.4` ✓），reset 与 step 的指标/info 键集合完全一致 ✓；已加入评估器白名单，会进 wandb 与 `progress.csv`。

### 9.28.3 视频/轨迹里的"哪只手"与"掉落是不是前提"

**渲染的 GIF 就是 run #6 的策略**（`runs/render/actor_latest/`，gif 21:21 / trace 21:20，run #6 结束于 19:27）。`trace.csv` 的事实（4 集）：

| episode | done@ | 脱离抓握窗 | 出现过的杆号 | 躯干 x 峰值 |
|---|---|---|---|---|
| 0–3 | 65–66 | L@3、R@4 | L=[−1,0,**1**]，R=[−1,0] (ep1–3 也有 1) | **+0.40**（B1 在 x=0.4556）@step 57 |

即：**两只手几乎立刻都脱离标定抓握窗**（步 3/4，属早前发现的 hook 模式），随后躯干前摆到 x≈0.40（离 B1 只差 5 cm），在 65 步左右掉落。**左手是那只靠近 B1 的手**（4/4 集）。

**"哪只手先伸"是学出来的，不是脚本定的**，依据：
* 模型/关键帧/目标都是左右镜像对称的（逐关节检查：pitch 同号、roll/yaw/wrist_roll 反号，腿部**完全**对称，臂部残差 ≤0.009 rad ≈0.5°；root y = −5 mm；goal 的 $p_L,p_R$ 对称）；
* 但 5 次旧 run（m1 窗）里**恰恰相反**：`bar_L≈0.00`（左手一直抓 B0）、`bar_R≈−0.95`（右手松开）⇒ 是**右手**在试探；
* run #6（reach 窗）里变成左手更常脱离 ⇒ **跨配置就换了手**，说明是策略自己学的分工；
* 但所有 run 都用 **`--seed 0`**（同一份网络初始化 + 同一串 reset 噪声 + 同一数据顺序），所以"同一配置下每次都是同一只手"很正常。要判定是否真有偏置，跑一次 `--seed 1` 看 `bar_L/bar_R` 是否互换即可。

⚠️ **相机视角容易看错左右**：`render_policy.py` 默认 `azimuth=70`（相机在 +y 侧，即机器人**左侧**），且 +x（前进方向）在画面里朝**左**——所以"看起来在伸的那只手"需要和 `trace.csv` 的 `c_L/c_R、bar_L/bar_R` 对照。已在渲染器的 trace 里新增 **`d_LB1` / `d_RB1`（每只手到 B1 的距离）**，并打印每集"离 B1 最近的是哪只手"，以后不用靠肉眼判。

**"先保住抓握"是必要前提吗？** 概念上不是——掉落轨迹里的"伸手状态"照样会作为重标记目标进入 buffer。但有一条**统计上的**硬约束：`flatten_batch` 只在**同一 traj_id 且更靠后**的帧里抽目标，所以长度 L 的轨迹提供 ~L²/2 个有效 (state, goal) 对（每步 ~L/2 个）。run #6 的 L≈60 **小于 `unroll_length=62`** ⇒ 每个 unroll 内部跨一次 reset，跨轨迹对全被 mask ⇒ 每个 transition 的正样本数掉约 8 倍，buffer 也被短命轨迹冲刷（critic `pos−neg` 13.3→6.9、`acc` 0.54→0.31 正是这个）。
⇒ 目标不是"永不掉落"，而是 **episode 明显长于 unroll（≥200–300 步，最好 ~500）**，同时保留伸手的探索。run #6 在 2.55 M 步那一窗（`len=496, fell=19%`）就是理想区间。

## 9.29 run #7（A2′：reach + `expl-hold 10`）结果 —— 够杆已解决，剩下的是"抓住并保住"

20 个 eval，12,197,632 步（`runs/brach_reach_b1_a2p/`）。覆盖率指标首次可用，结果比 `max_bar` 单独看清晰得多：

| steps | len | fell | 换手/步 | C(0.2) | C(0.1) | **最近距离** | 摆幅 | 支撑率 | max_bar(和) |
|---|---|---|---|---|---|---|---|---|---|
| 1.34 M | 501 | 0% | 1.00 | 0.00 | 0.00 | 0.374 | 0.067 | 0.97 | 0 |
| 3.75 M | 70 | 100% | 0.329 | 1.00 | 1.00 | **0.049** | 0.216 | 0.19 | 0.75 |
| 4.96 M | 67 | 100% | 0.179 | 1.00 | 1.00 | **0.022** | 0.279 | 0.08 | 2.62 |
| 8.58 M | 67 | 100% | 0.280 | 1.00 | 1.00 | **0.016** | 0.330 | 0.18 | 2.25 |
| 12.20 M | 68 | 100% | 0.308 | 1.00 | 1.00 | **0.015** | 0.330 | 0.15 | **4.25** |

（`最近距离 = 2.0 − eval/episode_cov_min_d_b1_improve`；`C(r)=eval/episode_cov_cross_b1_*`；`支撑率 = cov_hand_on_steps/len`）

**结论**：
1. **"够到下一根杆"已经解决**：`C(0.1)=1.00`（每个 episode 都进入 B1 的 10 cm 内），最近距离稳定在 **1.5–5 cm**（抓握判据 5 cm），摆幅从 0.067 长到 0.33 m（不再是不动）。这是六次 run 以来第一次。
2. **`expl-hold 10` 没能把 episode 拉长**（仍 ~67 步、100% 掉落），只把"稳定但不动"的那一窗推到了 1.34 M。
3. **卡点变了**：现在是"**冲过去碰一下、然后掉**"——`支撑率` 只有 0.08–0.19（任意时刻只有约 15% 的步数有手在抓握窗内），`max_bar` 的和只有 4.25（≈ 只有最后 4 步处于"运行最大值=1"）⇒ 碰到 B1 后立刻掉落、没保住。

**机制解释（重要）**：现用的 `support` goal 是 `[x_torso, z_torso, p_L(3), p_R(3), max(c_L,c_R)]`，其中
`c = exp(-(d_min/τ)²)` 用的是**最近那根杆**的距离，而 `p_L,p_R` 是**绝对世界坐标**。于是"飞行途中两手恰好掠过 B1 的悬挂位置 + 顺手被算成 c≈1"这一瞬间就足以把 goal 距离压到很小 ⇒ **这个 goal 可以被"一次性飞掠"满足**，正是执行计划 §1.4 警告过的退化解（"荡到目标附近就掉"）。`max(c_L,c_R)` 还把"哪只手抓哪根杆"完全抹掉。

**修法（已实现，就是诊断文档第 3 条）**：新增 `--goal-variant cross`：
$$g=\big[\,p_R-p_{B_1}\ (3),\ c_{R,B_1},\ c_{L,B_0}\,\big]\quad(5\ \text{维})$$
* `c_{R,B1}` / `c_{L,B0}` 是**指定手对指定杆**的接触量（不是"最近杆"）⇒ 飞掠式冲刺**不可能**满足（左手离开 B0 就把 `c_{L,B0}` 打到 0，距离 ≥1）；
* `p_R - p_B1` 是**相对目标杆**的位置 ⇒ 不再奖励"身体摆到某个绝对构型"；
* 两只手不再被 `max` 抹平。

实测标定（原生 MuJoCo 校验）：关键帧处 `achieved = [−0.400, 0, 0, 0, 1]`（右手离 B1 0.4 m、未接触 B1；左手在 B0 ⇒ `c_LB0=1`），`dist0 = 1.077`；理想双支撑态 `[0,0,0,1,1]` 距离为 0；**纯 B1 悬挂态距离 = 1.0（不算成功）** —— 这正是 M3 要的"转移态"，而不是"最终悬挂态"。四种 variant 的 obs 尺寸：`full 151 / support 151 / position 149 / cross 151`（互不影响，旧 checkpoint 仍可回放）。

## 9.30 cross goal 的修订：`cross3`（过渡态）与 `hold2`（终点态）

对 §9.29 里那版 `cross = [p_R−p_B1(3), c_{R,B1}, c_{L,B0}]` 的两条批评都成立，已改：

1. **`p_R` 是什么**：它是"**抓握座位点**"= `xpos[wrist_roll_link] + R_wrist · seat_local`，`seat_local` 是 M0/M1 用关键帧标定出来的常量，使这个点正好落在杆心 —— env 的 `_grasp`/`grasp_dist` 判据用的就是它（也就是 M3.0 脚本里那个点）。所以 `p_R−p_{B1}` 的物理含义是"**目标杆相对右手的向量**"（手落在 B1 上时为 0），既不是指尖/末端，也不是随手取的点。
2. **但它作为 goal 项确实冗余且过约束**：`c_{R,B1}=exp(−(d/0.04)²)` 在 d≲4 cm 就饱和，"接触"这一项已经表达了"手在 B1 上"；再叠 3 维位置（量纲 0.4 m，压过接触项）属于重复，而且把"手的绝对姿态"也带进了 goal。
3. **"身体到 B1 的距离 = 0"这个提法**：在"左手仍抓 B0"的约束下**不可达** —— B0→B1 相距 0.4 m，而挂在一只手上时躯干最多前移约 0.2 m（臂长 ~0.35–0.45 m、身体垂在手下方）。躯干真正位于 B1 下方只发生在**终点态**（双手都在 B1）。所以"身体位置"只适合放在**终点**目标里，放在**过渡**目标里要么不可达、要么变成"停在两杆正中间"这种魔法数字。

新增两个变体（与 `cross` 并存，`train.py --goal-variant`）：

| variant | goal | 维 | obs | 用途 / 代价 |
|---|---|---|---|---|
| `cross3` | `[d_{R,B1}, c_{R,B1}, c_{L,B0}]` | 3 | 152 | **过渡态**：把 3 维手姿换成**标量距离**（单调、平移不变、无绝对坐标、与饱和的接触项不重复），`c_{L,B0}` 保证"松手飞掠"不可行（松 B0 ⇒ 距离≥1）。标定：关键帧 `[0.400, 0, 1]` ⇒ `dist0=1.077`；过渡态 0；纯 B1 悬挂态 1.0（不算成功） |
| `hold2` | `[c_{L,B1}, c_{R,B1}]` | 2 | 151 | **终点态**（你说的"稳定停在第二根杆"）：纯接触、无位置、无需编码身体。关键帧 `[0,0]` ⇒ `dist0=1.414`；此时 `dwell_success`（0.5 s 内 dist<0.35 且低速）恰好等于"**双手都在 B1 上保持 0.5 s**"，是它的天然读数 |

`hold2` 的两个代价要记住：① **飞掠也能满足**（两手掠过 B1 的瞬间 c 都≈1）——所以要靠 `dwell_success` 而不是 `success` 来判"稳定停住"；② **只有 2 bit**，重标记出的 goal 只剩 ~4 种模式，ψ 几乎没东西可编码，对比学习的粒度会显著变差（这正是原诊断文档要留一个连续分量的原因）。因此建议：**过渡阶段用 `cross3`，终点阶段用 `hold2`**。

旧变体的 obs 尺寸逐项不变（full 151 / support 151 / position 149），旧 checkpoint 仍可回放。


### 9.30.1 启动崩溃：`eval_kwargs['goal_bar']`（已修）+ 第三道静态检查

cross 家族不再使用 `goal_bar`，但我在打印里仍直接索引 `eval_kwargs['goal_bar']` ⇒ 启动时 `KeyError`（env 已建好、训练未开始，无害）。

修法：**让 `goal_bar` 永远是已有键**（cross 家族置 `None`，反正 `reset()` 里对它短路），打印改用 `.get` 并按 variant 分支。

这类"只在某些路径存在的键被打印/调用读取"的崩溃已经出现三次（`args.use_relu`、`CRL(expl_hold=)`、`eval_kwargs['goal_bar']`），所以给 `src/check_args.py` 加了**第三道静态检查**：把所有 `env_kwargs['k']` / `eval_kwargs['k']` 的**读取**与"`dict(...)` 初始化键 + 无条件赋值"对照，未覆盖且未用 `.get` 的一律 FAIL。现在三条检查一起跑：

```
.venv-warp/bin/python src/check_args.py
OK   src/train.py  flags=41 used=41 missing=[]
OK   kwargs-dict reads: 3 read, unguarded = []
OK   CRL(...) kwargs: 19 passed, missing fields = []
```
（跑 GPU 前先跑这条，能挡住这一类"跑到一半才炸"的错。）

**run #8/#9 的命令不变**（`--goal-variant cross3` 与 `--goal-variant hold2`，其余同 run #7），见 §9.30。

## 9.31 run #8（`cross3`）结果 —— 收紧 goal 反而把 critic 弄崩了（**否证**：光改 goal 语义不够）

20 个 eval，12,197,632 步（`runs/brach_cross3_short/`）。命令同 run #7，只把 `--goal-variant support` 换成 `cross3`。

| steps | len | fell | 换手/步 | C(0.2) | **最近距离** | 摆幅 | 手在杆率 | bar_L(和) | bar_R(和) | max_bar | succ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.74 M | 7.3 | 100% | 0.68 | 0.00 | 0.240 | 0.080 | 0.54 | −2.4 | −5.6 | 0 | 0 |
| 3.75 M | 6.4 | 100% | 1.00 | 0.00 | 0.324 | 0.094 | 0.94 | 0.0 | −4.3 | 0 | 0 |
| 7.98 M | 13.0 | 100% | 0.99 | 0.00 | 0.242 | 0.069 | 0.97 | −0.1 | −11.5 | 0 | 0 |
| 12.20 M | 14.9 | 100% | 1.00 | 0.00 | **0.247** | **0.082** | 0.98 | 0.0 | −13.6 | 0 | 0 |

对照 run #7（a2p，同期）：`len 67`、最近距离 **0.015**、摆幅 **0.330**、`C(0.1)=1.00`。**cross3 是六次 run 里最差的一次**：episode 只有 5–17 步（a2p 的 1/5），自由手最近只到 24 cm（a2p 的 15 倍），摆幅退回 0.08 m（"稳定但不动"那一档）。

### 9.31.1 为什么收紧 goal 会让学习崩掉：3 维饱和 goal 让对比任务退化

训练侧指标（每 3 个 eval 取一个）：

| steps | cross3 acc | cross3 pos−neg | cross3 critic | a2p acc | a2p pos−neg | a2p critic |
|---|---|---|---|---|---|---|
| 0.74 M | 0.020 | 2.01 | 5.62 | 0.368 | 6.91 | 3.57 |
| 2.55 M | **0.086** | **5.01** | 4.45 | **0.455** | **7.83** | 2.67 |
| 6.17 M | 0.110 | 4.12 | 4.50 | 0.407 | 7.89 | 2.85 |
| 12.20 M | 0.083 | 2.54 | 5.04 | 0.257 | 3.91 | 4.20 |

**cross3 的 `categorical_accuracy` 始终在 0.08–0.11**（a2p 峰值 0.46），`critic_loss` 一直在 4.4–5.6 高位（a2p 最低 2.67）。也就是说：**critic 从来没有学出一个可用的度量**，actor 自然拿到的是噪声梯度（`actor_loss` 反而更高）。

机制：`cross3 = [d_RB1, c_RB1, c_LB0]` 三个分量里
* `c_RB1`、`c_LB0` 用 `exp(−(d/0.04)²)`，在 d≳6 cm 就**饱和到 0/1**，几乎不携带"离得多近"的信息；
* `d_RB1` 在整个"左手抓 B0"的构型空间里只从 0.35 变到 0.24（**总共 ~11 cm 动态范围**，而 a2p 的 8 维绝对位置有 ~0.4 m 的连续谱）；
* 两者叠加的结果是：一集之内 `achieved` 几乎不变 ⇒ **"未来状态"作为 goal 时，goal 与当前状态的距离天然就很小** ⇒ 对比任务近乎退化（正负样本在 3 维空间里挤在一起）⇒ acc≈0.1、pos−neg 2.5。

反过来看 a2p：那 8 维绝对位置（$x,z,p_L,p_R$）虽然"语义上不该进 goal"，却**正是提供连续谱、让 critic 有东西可编码的那一项**；策略把"身体摆到 B1 附近的某个绝对构型"当成可达的子目标，于是学会了摆荡+伸手（最近 1.5 cm）。§9.29 里"`max(c_L,c_R)` 太松"的批评成立，但**修法不能只是删维度**。

### 9.31.2 结论：goal 语义不是当前的第一杠杆，**几何**才是

* 6 次 run 的唯一一次"够到杆"（a2p）发生在 `reach` 窗 + `support` goal + 0.40 m 杆距；
* 它的失败模式是**掠过后掉**（`bar_R` 只在最后 ~4 步非负、支撑率 0.15）；
* M3.0 已经量过：0.40 m 时静态极限就是 0.031 m（3.1 cm，勉强进 5 cm 判据）；**动态/带 PD 误差时几乎必然差几厘米**。于是"掠过而不握住"正是几何给出的必然结果，而**不是**探索或 goal 的锅。

⇒ 下一步（本条已实现，见 §9.32）：**只改一个变量——把杆距从 0.40 m 改成 0.35 m**，其余完全照抄 run #7。M3.0 的静态极限在 0.35 时是 **0.0007 m（0.7 mm）**，动态余量从"差 3 cm"变成"富余 4–5 cm"。

`hold2` **不跑**：它只有 2 bit（4 种模式），按 9.31.1 的机制只会比 cross3 更退化；保留为**评估用**的终点态定义（`dwell_success` 天然就是"双手在 B1 上停 0.5 s"）。

## 9.32 M3.1 第一步：0.35 m 几何资产 + 双支撑指标（run #10 待跑）

### 9.32.1 新资产 `--scene full035`

用**与 `scene_bars.xml` 完全相同的 build 参数**、只把 `spacing` 换成 0.35 导出（`assets/g1_brachiation/build_scene.py`）：

```bash
.venv-warp/bin/python assets/g1_brachiation/build_scene.py \
  --export assets/g1_brachiation/scene_bars_d035.xml --spacing 0.35
```

（等价于 `bs.export_scene(bs.SceneParams(spacing=0.35, mjx_compat=False), path=...)`。`build_scene.py` 的 `--export/--spacing/--mjx-compat` 这条 CLI 就是本次加的；已实测重跑导出与现有文件**逐字节相同**，即这条命令是资产的可复现来源。）

实测（原生 MuJoCo 读回 XML，与 0.40 版逐项对照）：

| | `scene_bars.xml`（0.40，全部旧 run） | `scene_bars_d035.xml`（0.35，新） |
|---|---|---|
| `bar_x` | 0.0556 / 0.4556 / … | 0.0556 / **0.4056** / … |
| 杆距 | 0.4000 | **0.3500** |
| `nq / ngeom / nkey` | 50 / 105 / 2 | **50 / 105 / 2** |
| 关键帧 `z / hip_pitch` | 0.7743 / −0.6378 | **完全相同** |
| 静态 IK 极限 $\lVert p_R-p_{B_1}\rVert$ | 0.0310 m | **0.0007 m** |
| 关键帧按住 1 s | $\Delta z=-5.2$ mm | $\Delta z=-5.2$ mm（逐位相同） |

**关键性质：`bar_x[0]` 与关键帧完全不变**（挂钩姿态是从 B0 解出来的，B0 没动）⇒ 初始状态、抓握标定、goal 集合的**生成方式**都不变，唯一变化是"下一根杆近了 5 cm"。这是一次干净的**单变量**改动。

注册：`--scene full035`（`train.py` / `check_reset.py` / `smoke_env.py` / `goal_geometry.py` / `render_policy.py` 全部可用）。obs 维度不变（`full` 151 / `support` 151 / `position` 149）⇒ **旧 checkpoint 仍可回放**，新 checkpoint 也会把 `scene=full035` 写进 config，渲染器自动用它。

上表的两个数都可以在原资产上复现（`m3_swing.py` 本次加了 `--scene`，因此不必再靠内存构建）：

```bash
.venv-warp/bin/python src/m3_swing.py --scene full035 --ik
# [scene] scene_bars_d035.xml (exported XML)  spacing=0.3500 m ...
# [ik] p_R = (+0.4063, -0.1729, +1.4857) -> |p_R - B1| = 0.0007 m  (B1 at x=0.4056)
# [ik] left hand still at |p_L - B0| = 0.0000 m
.venv-warp/bin/python src/m3_swing.py --scene full035   # 动态/脚本化模式同 0.40 版
```

⚠️ **一处需要记住的次生效应**：`goal_reach_thresh` 仍是 **0.35 m**（绝对值，没跟着杆距缩放）。一面杆的步长是 $\sqrt3 d$：0.40 时 0.6928 m ⇒ 阈值 = 0.505 根杆；0.35 时 0.6062 m ⇒ 阈值 = **0.577 根杆**，即 `success` 稍宽松。为了保证"只改几何"这一次仍是单变量，**这次不动它**；若要恢复"半根杆"语义，加 `--goal-reach-thresh 0.303`（下次要调时再动）。

### 9.32.2 新指标：双支撑 / 换手 / 滞空时长（M3.1 补完，纯记录）

`_coverage()` 新增（与 `bar_L`/`bar_R` 同一判据：某手最近杆距离 < 0.05 m；`bar 0` = 出发杆，`bar 1` = 下一根）：

| 指标 | episode 求和 = | 读法 |
|---|---|---|
| `cov_both_on_steps` | 双手同时抓**任意**杆的步数 | 除以 `avg_episode_length` = 双支撑占比。⚠️ **会被"开局双手都挂 B0"这一段灌满**（run #10 实测 `cov_both_runmax` 14–19 全是开局那 15 步），不能当"抓住下一根杆"的证据 |
| `cov_both_runmax_improve` | 上面的**最长连续段**（步） | 同上，开局即饱和 |
| `cov_b1_runmax_improve` | **双手都在 B1 上的最长连续段**（步） | ≥25 步 = 0.5 s = 一个 `dwell_steps` ⇒"抓住下一根杆并保持" |
| `cov_xfer_runmax_improve` | **一只手在 B0、另一只手在 B1** 的最长连续段（步） | 真正的**换手过渡态**（左右通用）；恒 0 = 策略从不让一只手留在旧杆上 |
| `cov_air_runmax_improve` | **双手都不在杆上**的最长连续段（步） | "滞空/鱼跃"时长；run #10 实测 ≈11 步（0.22 s） |

实现方式与已有覆盖率指标一致（计数器只增不减，"改进量"在求和型评估器下精确还原最大值）。**不改物理、不改 obs/goal**（只加计数键与 `info` 里的计数器），旧 checkpoint 不受影响。

三根计数器已在 CPU 实跑校验（`--scene full035`，HOLD 动作 30 步）：`cov_both_run=30`、`b1/xfer/air=0` ✓（换手/滞空计数器在 HOLD 下必须为 0）。

新增静态检查 `src/check_metrics.py`（防止"指标写了没初始化 / 没进评估器白名单"这类静默丢诊断的问题；现在也会检查 `streak(cond, "cov_..._run")` 这种间接读取的计数器键）：

```
.venv-warp/bin/python src/check_metrics.py
metrics emitted by _coverage : 14
  cov_air_runmax_improve, cov_b1_runmax_improve, cov_both_on_steps, ... , cov_xfer_runmax_improve
info keys read in reset/step : 8 (cov_air_run, cov_b1_run, cov_both_run, ...)
OK  metric/info keys are consistent and all metrics are logged
```

跑 GPU 前建议与 `check_args.py` 一起跑。

### 9.32.3 run #10 命令（1.5 h 探针；与 run #7 只差 `--scene`）

```bash
.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant support --train-goal-bar 1 --eval-goal-bar 1 \
  --expl-hold 10 --num-evals 20 --steps 12200000 \
  --checkpoint-dir runs/ckpt_d035 --save-every 5 \
  --impl warp --scene full035 --wandb --exp-name brach_d035_reach
```

跑前：`.venv-warp/bin/python src/check_args.py && .venv-warp/bin/python src/check_metrics.py`

**判据（看 `cov_both_runmax_improve`，这是这次真正的新信息）**：

| 现象 | 判读 | 下一步 |
|---|---|---|
| `len` 仍 ~60–70、`fell` 100%、`cov_both_runmax` ≤ ~5 | 几何不是瓶颈，0.35 只是让"掠过"更近 | 降噪声（`--entropy-param 1.6`，§9.28.1 A2）或继续降到 0.30（§9.32.1 同一套命令只换资产） |
| `cov_both_runmax` 涨到 20–40、`len` 涨到 100–200 | 抓住并短暂保持出现了 | 直接跑满 4 h，看 `len` 是否继续长、`forward`/`cov_fwd_improve` 是否开始动 |
| `len` 回到 ~500、`cov_both_on_steps/len` > 0.5 | 双支撑稳定，只是还没向前转移 | 进 M3.2（把"转移"写进 goal：`cross3` 的连续分量要重新设计，见 9.31.1） |

**注意**：`success`/`max_bar` 这次会**更容易**满足（阈值占比变宽 + 杆更近），所以判"是否真的抓住"以 `cov_both_runmax` 与 `bar_R`/`cov_hand_on_steps` 为准，不要只看 `success`。

## 9.33 run #10（0.35 m 几何）结果 —— **够杆已彻底解决，但策略学成了"双手离杆鱼跃"**

命令与 run #7 逐项相同，只把 `--scene full` 换成 `--scene full035`（`runs/brach_d035_reach/`，20 evals，12,197,632 步）。

| steps | len | fell | C(0.2) | C(0.1) | **最近距离** | 摆幅 | 手在杆率 | both | bothmax | **B1 双持**≥ | **换手 L0/R1** | **滞空** | bar_L | bar_R | max_bar | succ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.74 M | 208 | 69% | 0.00 | 0.00 | 0.219 | 0.087 | 0.55 | 2.6 | 2.6 | — | — | — | −205 | −62 | 0.00 | 0.00 |
| 1.94 M | **501** | **0%** | 0.00 | 0.00 | 0.238 | 0.064 | 0.97 | 6.6 | 5.8 | — | — | — | −494 | −8.9 | 0.00 | 0.00 |
| 3.15 M | **501** | **0%** | 0.00 | 0.00 | 0.236 | 0.111 | 0.53 | 12.6 | **12.1** | — | — | — | −477 | −188 | 0.00 | 0.00 |
| 4.96 M | 30.8 | 100% | 1.00 | 1.00 | 0.038 | 0.189 | 0.48 | 8.8 | 6.5 | — | — | — | −13.2 | −14.7 | 2.31 | 0.25 |
| 7.98 M | 41.1 | 100% | 1.00 | 1.00 | **0.006** | 0.291 | 0.65 | 23.7 | 17.4 | — | — | — | −1.9 | −9.2 | 9.19 | 6.12 |
| 12.20 M | 40.9 | 100% | 1.00 | 1.00 | **0.006** | 0.280 | 0.60 | 21.0 | 14.3 | — | — | — | −3.5 | −9.6 | **12.12** | **7.12** |

（`both`/`bothmax` 是本次加的"任意杆双手支撑"指标；后三列在 run #10 时还没有，事后用 trace 补算，见 9.33.2）

**对比 run #7（a2p，同期 12.2 M 步）**：

| | run #7（0.40 m） | **run #10（0.35 m）** |
|---|---|---|
| 自由手最近距离 | 0.015 m | **0.006 m** |
| `max_bar` 和（≈ 在 B1 窗内的步数） | 4.25 | **12.12** |
| `succ`（每集 `dist<0.35` 步数） | 1.75 | **7.12** |
| 每步平均 `dist`（`episode_dist/len`） | 1.32 | **0.67** |
| `C(0.1)` | 1.00 | 1.00 |
| `len` / `fell` | 67 / 100% | 41 / 100% |
| 摆幅 | 0.330 | 0.280 |

**几何的预期效果全部兑现**：最近距离 1.5 cm → **6 mm**，真的有一步 `dist = 0.06 m`（= B1 悬挂构型本身，不只是"手靠近"），B1 窗内步数 4 → 12。**"够杆"这个问题可以结案了。**

### 9.33.1 两个阶段 + 失败的机制（用 CPU 回放逐步查清）

1. **0.74–3.15 M：先学会"单手挂住不动"**——`len` 到 501、`fell=0`，但 `bar_L≈−477`、`bar_R≈−188`（左手一直脱离、右手挂在 B0 上）、摆幅只有 0.06–0.11 m。即又回到"稳定但不动"那一档（无害，但也不是技能）。
2. **3.75 M 起主动放弃稳定，换"鱼跃"**——`len` 掉到 ~40、`fell` 到 100%，但最近距离、`max_bar`、`succ` 一路涨到上面的值。

**回放**（本机 CPU 上跑通了 MJX-Warp！见 9.33.3）：最终 checkpoint、`--goal-bar 1`、4 集 × 160 步。

| 集 | 双手挂 B0 | 躯干 z 峰值 | **离杆滞空** | 双手落 B1 的步数 | **换手态(L0&R1)** | 落地 $v_z$ | 掉落@ | 掉落时 z |
|---|---|---|---|---|---|---|---|---|
| 0 | 16 步 | 0.910 | **11 步 (0.22 s)** | 5 | **0** | −1.13 m/s | 34 | 0.663 |
| 1 | 15 | 0.903 | 10 | 5 | **0** | −1.11 | 35 | 0.684 |
| 2 | 16 | 0.911 | 12 | 8 | **0** | −1.00 | 35 | 0.679 |
| 3 | 15 | 0.905 | 11 | 9 | **0** | −0.98 | 49 | 0.731 |

逐步过程（每集都一样）：**①** 双手挂 B0，躯干从 0.774 被"引体向上"拉到 **0.90（+13 cm）**；**②** **双手同时松开**，滞空 0.2–0.24 s、飞越 0.35 m；**③** 从**上方**落到 B1，落地竖直速度 **−1.0 … −1.13 m/s**；**④** 双手在 B1 上撑 **5–9 步（0.1–0.18 s）**，最好的一步 `dist = 0.06`；**⑤** 滑脱，z ≈ 0.66–0.73，自由落体（一直掉到 −27 m ⇒ `done` 是**真掉落**，不是摆动把 z 压过 `FALL_DEPTH` 阈值，这个疑点排除）。

**结论（诊断）**：

1. `support`/`full` 的 goal 是**整机构型目标**。"身体处于 B1 的悬挂构型"这一项**只有把整个身体送过去**才满足；而 M3.0 证明的静态解（左手留 B0、右手伸到 B1，dist=0.0007 m）在这个 goal 下只解决了右手那 3 维，剩下（躯干 $x,z$ + 左手）还差 **0.606 m**（= $\sqrt3\,d$，见 §9.32 的 goal 几何）。
2. 因此**最短路径就是松手鱼跃**：0.2 s 把整机构型送到 `dist=0.06`。CRL 没有能耗/时间代价，所以这条路径在 goal 意义上**是最优的**——策略没有"学错"，是 goal 让它这么学的。
3. 但物理兑现不了：落地竖直动量 $34.4\times1.1\approx38\ \mathrm{N\cdot s}$，要在 ~0.1 s 里被手臂吸收 ⇒ 需要数百牛的接触力/相应的肩力矩。仿真里偶尔成功（ep 3 撑了 ~20 步），多数时候"碰一下就滑"。
4. 也就是说：**现在的卡点不是"够不到"、也不是"抓不住"，而是 goal 只表达"到达哪个构型"，不表达"支撑是否连续、落点速度是否可兑现"。** 这正是执行计划 §1.4 警告的退化解，现在有了逐步证据。
5. 另外：**策略从不做真正的换手**（`(左手 B0, 右手 B1)` 四集全为 0），它是"双手一起跳过去"。

### 9.33.2 指标 bug 修正：`cov_both_runmax` 其实在量"开局的 B0 双持"

run #10 的 `bothmax` 14–24 步**全部来自开局双手都挂 B0 的那 15–16 步**（trace 实测），与"抓住下一根杆"无关。已换成三根**与杆编号绑定**的计数器（纯记录，不改 obs/goal）：

| 新指标 | episode 求和 = | run #10 事后实测 |
|---|---|---|
| `cov_b1_runmax_improve` | **双手都在 B1 的最长连续段**（步） | **4–7 步**（0.08–0.14 s；≥25 步才算"保持一个 dwell"） |
| `cov_xfer_runmax_improve` | **一手 B0 + 一手 B1** 的最长连续段（左右通用） | **0**（从未换手） |
| `cov_air_runmax_improve` | **双手都不在杆上**的最长连续段 | **≈11 步**（鱼跃时长） |

这三根已在 CPU 实跑校验（HOLD 动作 30 步：`both_run=30`、其余全 0 ✓），并接入 `src/check_metrics.py` 的静态检查（14 个指标 / 8 个计数器键全部一致）。

### 9.33.3 顺带解锁：本机 CPU 能跑 MJX-Warp（回放不需要 GPU）

`JAX_PLATFORMS=cpu` 下 `mjx.put_model(impl="warp")` + `mjx.step` **可以正常编译和执行**（2 env 首次编译 ~13 s，4 env × 160 步 + 渲染 ~1 min），于是策略回放/视频/逐步诊断在本地就能做：

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl .venv-warp/bin/python src/render_policy.py \
  --ckpt runs/ckpt_d035/actor_latest.pkl --episodes 4 --steps 160 --goal-bar 1
# -> runs/render/d035_final/{trace.csv, frames/*.png, d035_final.gif}
```

`trace.csv` 现在能回答"哪一步在杆上、哪一步离杆、落下多快"——以后不必等 GPU 排队就能先看行为，再决定要不要花 4 h。

### 9.33.4 下一步（三选一，推荐 C）

| | 做法 | 为什么可能有效 | 风险 |
|---|---|---|---|
| **A** | `--goal-variant cross`（5 维 `[rel_p_R(3), c_RB1, c_LB0]`，**从未跑过**；run #8 跑的是 cross3） | 有静态最优解（M3.0 的 IK，dist→0）；`c_LB0` 把鱼跃**直接判死**（松了 B0 ⇒ 距离 ≥1）；`rel_p_R` 有 0.35 m 连续谱（§9.31.1 的教训满足） | 只奖励"换手过渡态"，教出来的是"抓住 B1、身体还在 B0"= M3 的**前半段**；而且它要求"右手先上"（实测策略是左手先到位，可能要多花时间换基） |
| **B** | 任务定义修正：**双手离杆 ≥k 步即 `done`**（"brachiation 必须至少一只手在杆上"） | 直接把鱼跃从 replay 里删掉，逼出"另一只手先抓稳再松手" | episode 会先变得很短（现在 16 步就离杆）⇒ §9.28.3 的 $L\ll$ unroll 统计问题加重 |
| **C**（推荐） | `support` 加 1 维**持续接触**：`hold = 1 − exp(−streak/10)`，`streak` = 连续 `c_sup>0.5` 的步数（闸门放 `info`） | 8 维位置 + 连续谱全部保留（避开 §9.31.1 的退化）；但"碰到就走"不再满足 goal：撑 5 步 hold≈0.39、撑 25 步 0.92。eval `success` 随之变成"到位**并保持**"，与 M1/M3 的验收口径一致 | hold=1 的 B1 状态初期稀有（实测最长 4–7 步）⇒ 必须做**平滑**的 hold（不能是 25 步硬阈值），否则梯度太稀 |
| D | 什么都不改，跑满 4 h | — | 最后 4 M 步全部指标已平（succ 6–7、最近 0.006–0.012），鱼跃是稳定吸引子 ⇒ 预期收益低，**不建议** |

推荐 **(C)**，若 (C) 之后 `cov_air_runmax` 仍 ~11 且 `cov_b1_runmax` 不涨（说明策略宁可选鱼跃也不肯保持），再上 **(B)** 把滞空直接禁掉。

## 9.34 M3 收尾：`support_hold`（把"持续接触"写进**目标状态**）+ 修掉一个一直在骗人的指标 bug

### 9.34.1 ⚠️ 先修 bug：`max_bar` 与 `dwell_success` 从第一版起就是错的

`_coverage()` 以前返回 `dict(info)`（整份 info 的拷贝 + 覆盖率键），而 `step()` 把它作为**最后一次** `info.update()` 应用 ⇒ 它把 `max_bar`、`dwell`、`prev_bar`、`switches_total` **回退成上一步的值**。后果：

| 量 | bug 期间的真实含义 | 应有含义 |
|---|---|---|
| `dwell` | 永远 ≤1（每步都被回退成 0）⇒ **`dwell_success` 在任何 run 里都不可能为 1** | 连续 25 步"成功且慢" |
| `max_bar` | **当前这一步**的杆号（不是 episode 最大值） | 本 episode 抓握过的最大杆号 |
| `prev_bar` | 上上步的杆号 ⇒ `hand_switches` 是"与 2 步前比"（偏小） | 与上一步比 |
| `switches_total` | 从不累加 | 累加 |

所以：**旧 run 表里的 `max_bar` 要读作"这一步手在 $B_k$ 上"的逐步求和**（§9.31–9.33 的表就是这么用的，结论不受影响）；而 **`dwell_success` 在所有旧 run 里恒 0 是 bug，不是策略行为**。

修法：`_coverage()` 只返回覆盖率键（不再拷贝整份 info）+ `step()` 把权威值放到**最后一次** update。覆盖率指标本身一直是对的（它们的键只在 `_coverage` 里更新）；M1 的 `check_reset` 用直接算出的抓手判据，也不受影响。

### 9.34.2 新变体 `support_hold`（"M3 收尾"这一枪）

$$g=\underbrace{[x,z,p_L(3),p_R(3),\max(c_L,c_R)]}_{\text{= support}}\;+\;\underbrace{\big[\,1-e^{-\text{streak}/10}\,\big]}_{\text{hold}}$$

`streak` = **连续**"至少一只手在任意杆 5 cm 抓握窗内"的步数；一旦两手都出窗立刻归零。

- 位置 8 维与连续谱全部保留（避开 §9.31.1 的退化），只多 1 维、尺度 0–1、平滑（5 步 0.39 / 10 步 0.63 / 25 步 0.92）——**不是 25 步硬阈值**，否则 run #10 实测只有 4–7 步，梯度会饿死。
- **这就是"不限制策略、只设计目标状态"**：鱼跃依然允许，但它的终点不再是目标状态。run #10 回放里策略最好的一刻是 `dist=0.06`（位置维已到位），而那一刻 `hold≈0`（两手刚离杆/刚碰杆）⇒ 加了 hold 后**该时刻的 `dist≈1.0`**；真的抓住 $B_1$ 并撑住 25 步 ⇒ `hold≈0.92` ⇒ `dist≲0.1`。
- `success = dist < 0.35` 因此变成"**到位 且 保持约 0.3–0.5 s**"，与 M1/M3 的验收口径（吊挂保持一段时间）一致。`dwell_success`（修好后）是它的天然读数。
- ⚠️ `streak` 用 5 cm 判据（`GRASP_THRESH`，与 `bar_L/bar_R/max_bar` 同源），**不是** `_coverage` 里的 `max(c)>0.5`（3.3 cm）：实测稳定悬挂时抓手点会从杆心漂到 **3–4.5 cm**（"钩住"状态），用 3.3 cm 会把经过 M1 验证的正常悬挂判成"没抓住"。

**CPU 数值校验**（`--scene full035 --goal-bar 1`，`envs=2`）：

| 检查 | 结果 |
|---|---|
| 布局 | `state=143 / goal=10 / obs=153`，`goal_indices=(0,2,99..104,127,128)` ✓ |
| goal_set | 每根杆 `c_sup=1.00, hold=1.00`，`x` 按杆距平移 ✓ |
| reset 后两手 | `dmin = 0.0015 / 0.0033 m`（远离 5 cm 边界，属正常"埋进指笼"） |
| HOLD 30 步 | `streak 1→5→10→25→30`、`hold 0.095→0.393→0.632→0.918→0.950` ✓ 与设计表一致 |
| 同 30 步 | `max_bar=0`（正确：一直在 $B_0$）、`dwell=0`、`switches_total=1`、`dist 1.17→0.73`（只剩位置维 + `c_sup≈0.6` 的残留）✓ |

obs 153 ⇒ **与旧 checkpoint 不兼容**（新跑，不能续训）。

### 9.34.3 run #11 命令 + 判据

```bash
.venv-warp/bin/python src/check_args.py && .venv-warp/bin/python src/check_metrics.py

.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant support_hold --train-goal-bar 1 --eval-goal-bar 1 \
  --expl-hold 10 --num-evals 20 --steps 12200000 \
  --checkpoint-dir runs/ckpt_hold --save-every 5 \
  --impl warp --scene full035 --wandb --exp-name brach_hold_b1
```

| 现象 | 判读 | 下一步 |
|---|---|---|
| `cov_air_runmax` 仍 ~11、`cov_b1_runmax` 3–7、`succ` 掉到 ~0 | 策略仍偏好鱼跃（现在不达标），只是还没学会软着陆 | `--entropy-param 1.6` 降噪；或换 `cross` 目标（§9.33.4 A） |
| `cov_air_runmax` 下降、`cov_b1_runmax` ≥25、`dwell_success` > 0 | **"抓住并保持"出现了** = M3 单杆闭环 | 跑满 4 h + 出视频，然后进 M4 |
| `len` 回到 300–500 且 `dist` 稳在低位 | 已能稳定停在 $B_1$ | 直接 M4（设计草案见 `docs/执行计划.md` 的 M4 节） |

⚠️ 因为改了指标语义，本 run 的 `max_bar`/`dwell_success`/`hand_switches` **与旧 run 不可直接比**；`len`/`fell`/`succ`/`C(r)`/`cov_*` 仍可比。

### 9.34.4 代码版本管理（本轮同时做的工程改动）

从这一轮起项目根目录是 git 仓库（`main` = run #10 结束时的工作树快照）：

- 每个实验一个分支，例如本次 `m3/hold-goal`；`git checkout main` 即可回到上一次的代码，`git log --oneline` 看全部版本；
- 每次跑 GPU 前 `python src/train.py ...` 会把 **`git_commit`（含 `-dirty` 标记）写进 `args.json` 与 checkpoint 的 config** ⇒ 任何一个结果都能回溯到确切的代码版本；
- `third_party/jaxgcrl` 是上游浅克隆、被 `.gitignore` 排除，**我们对它的改动由 `patches/jaxgcrl_crl_losses.patch` 版本化**——本轮发现该 patch 已经过期（评估器白名单后来加过键），已用新增的 **`tools/make_patch.sh`** 重新生成并校验（它会 `git apply -R --check`，对不上就报错），以后改 jaxgcrl 必须跑它。

### 9.34.5 `--smoke` 配置的 `critic_loss=nan` 是**老问题**（A/B 证据），不是本轮改动引入的

`--smoke`（4 envs / batch 100 / min_replay 50 / episode 101 / unroll 20）跑 `support_hold` 时 `critic_loss/actor_loss/logits_*` 全为 nan。做了三组 A/B（同样的 flag，只换一个变量）：

| 代码 | 场景 | goal | 结果 |
|---|---|---|---|
| 本轮分支 | `full035` | `support` | **nan** |
| 本轮分支 | `full` | `support_hold` | **nan** |
| **baseline `52b2c3e`（`git worktree` 检出、完全未改）** | `full035` | `support` | **nan** |

⇒ **与 `support_hold` 无关、与 `full035` 无关、与本轮的指标修复无关**：tiny smoke 配置本身就会发散（4 envs、50 条 transition 就开始训练，`l2` 能量在 float32 下 `-inf`，`diag - logsumexp` 变 `inf - inf`）。真实配置不会（run #7–#10 共 12M 步的 `critic_loss` 一直是 3.5–5.0 的有限值）。

`--smoke` 仍然是**接线测试**（能验 obs/goal 维度、指标键、checkpoint 落盘），但**不能当数值健康检查**。

顺带加了一个 **NaN 守卫**：连续 3 个 eval 的 `critic_loss`/`actor_loss`/`logits_pos` 出现 nan 就打印并 `SystemExit(2)`（保留已写出的 checkpoint），免得 1.5 h 的 GPU run 白跑。

## 9.35 M4.0：平移不变的"推进一根杆"目标（`--goal-variant advance`）

### 9.35.1 run #11（`support_hold`）的结论 + 行为逐步分解

20 个 eval，12,197,632 步（`runs/brach_hold_b1/`，git `af945f8`）。**`hold` 一加进去效果就变了**：

| steps | len | fell | C(0.1) | 最近距离 | 摆幅 | b1 双持步数 | 离杆最长 | bar_L | bar_R | succ | dist |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.94 M | 501 | 0% | 0.00 | 0.272 | 0.067 | — | 0 | −481 | −496 | 0 | 731 |
| 6.77 M | 59.6 | 100% | 1.00 | 0.010 | 0.314 | 11.1 | 14.4 | −25 | −27 | 3.2 | 65 |
| 7.98 M | 190.8 | 81% | 1.00 | 0.003 | 0.533 | 66.9 | 16.0 | +38.8 | +10.8 | 55.3 | 131 |
| **11.59 M** | **367.6** | **31%** | 1.00 | 0.005 | 0.451 | **130.6** | 18.9 | **+270.9** | −104.3 | **178.2** | 188 |
| 12.20 M | 306.2 | 44% | 1.00 | 0.008 | 0.420 | 95.4 | 16.1 | +222.5 | −115.2 | 108.1 | 175 |

指标一路在涨、**没有平台**（len 190→368，succ 55→178）。把最终 checkpoint 在 CPU 上回放（`runs/render/hold_final/`，3 集 × 400 步）：

| 集 | 掉落@ | 左手在 B1 步数 | 右手在 B1 步数 | 双持最长 | 双手同时（任意杆）最长 | 离杆最长 |
|---|---|---|---|---|---|---|
| 0 | 45 | 24 | 5 | 2 | 4 | 357 |
| 1 | 未掉 | **349 / 400** | **133** | **52** | 52 | 18 |
| 2 | 43 | 13 | 15 | 7 | 366→（掉落） | 366 |

逐步过程：**①** 双手挂 B0 → **②** 松手鱼跃 0.2 s（和 run #10 一样）→ **③** 到 B1 时**两手都曾摸到**（如 ep0 step 21：`d_LB1=0.010, d_RB1=0.015`，`barL=barR=1`）→ **④** 右手 3 步内滑脱，**左手单独拽住 B1** → **⑤** 身体绕左手继续前摆，有时能保持 300+ 步（ep1），有时 B1 也滑脱就掉（ep0/2）。

### 9.35.2 是 goal 的问题还是步数的问题？——**goal 是绑定约束**

- **不是够不到**：右手确实摸到了 B1（15 mm，`barR=1`），说明手臂可达、动作窗够大。
- **是 goal 没有"要求"它留下**：`support_hold` 的接触项是 `max(c_L,c_R)`、`hold` 是"**至少一只手**在窗内"。左手一拽住 B1，这两项就都满分了 ⇒ 只剩 `p_R`(3/10 维) 在推右手，而它已经能达到（瞬时）。
- **更糟的是 self-curriculum 会把它锁死**：CRL 的训练目标是**策略自己未来帧的状态**（§1 C3）。一旦"左手拽住、右手悬着"成为多数未来状态，重标记出来的 goal 就大多是这种状态 ⇒ 策略被推向"完美地单手吊着"，而不是"把另一只手也放上去"。这是典型的 CRL 侧吸引子。
- **步数确实也没跑完**：指标到 12.2 M 仍在涨（没有任何平台迹象），所以再跑 4 h 会更好——但**在同一个侧吸引子里更好**（单手吊得更稳），不会自己长出第二只手。⇒ 要解决"第二只手"，必须让 goal 对它提要求；**M4 的目标形式正好天然满足这一点**（下一条）。

### 9.35.3 `advance` 目标：一根杆的"平移不变"目标（M4 的地基）

$$g=\big[\underbrace{x_{\rm com}-x_{B_{k+1}}}_{\Delta x},\;\underbrace{z_{\rm com}-z_{\rm hang}}_{\Delta z},\;\underbrace{d_{L,B_{k+1}}}_{\text{左手到下一根杆}},\;\underbrace{d_{R,B_{k+1}}}_{\text{右手到下一根杆}},\;\underbrace{1-e^{-\text{next\_streak}/10}}_{\text{hold\_next}},\;\underbrace{k_{\rm ref}-k_{\rm start}}_{\text{progress}}\big]\quad(6\ \text{维})$$

> `margin`（力矩余量）已按用户要求**从 goal 中删除**（"本意是想得到不同于鱼跃的策略，但不是 M4 的重点"）。它作为"省力/留电量"的想法保留在 git 历史（`def6286`）里，将来若做效率研究可以从那里取回；当前目标是纯粹的"推进一根杆"。

`progress` = 本 episode **已经推进了几根杆**（原始计数，1 根 = 距离尺度上的 1.0，与一个接触项同量级）。加它的原因见 §9.37：goal 完全平移不变时，"稳定吊在 $k$ 号杆"与"再往前一级的稳定吊着"在相对量上**长得一样**，重标记出来的大多数 goal 会被平凡满足 ⇒ 没有东西推着策略继续走。

**`hold_next` 是必需的，不是可选项**（run #11 的教训，见 §9.36）：前四个量都是**瞬时**的，"从下一根杆的悬挂位姿里飞掠而过、两手顺便擦到"就能在几帧内把距离压小——这正是 run #10 的退化解。`hold_next` 只在**手持续留在下一根杆窗内**时才涨，所以目标状态是"**到了而且留下了**"。

- `k_ref` = **最近一次"持续抓稳"（连续 ≥ `K_SUSTAIN=3` 步在 5 cm 窗内）的杆号**，存在 `info` 里（进 obs 作上下文，**不进 goal**）⇒ 滞空期间 `k_ref` 不动，"下一根杆"始终有定义，**鱼跃仍然可以表达**。
- 平移不变（Δx/Δz 都是差）+ 进度不变（参考杆跟着走）⇒ **同一个 goal 在每一根杆上都成立**。实测：从 B0 或 B1 出发，初始 `dist` 都是 **1.456** ✓
- **两只手的接触项分别针对"下一根杆"** ⇒ 单手（无论哪只）只能把 `dist` 压到 ~1.0，**单手解不再是目标状态** ⇒ 9.35.2 的侧吸引子被结构性消除（这是 `advance` 相对 `support_hold` 的关键差别）。
- `margin`（用户提的"留电量"）= 12 个臂/腰关节的**力矩余量**，纯状态函数（`qfrc_bias` / `jnt_actfrcrange`）。关键帧悬挂时 margin = **0.854**（还剩 85% 余量），勉强抓住/撑住时会掉下来 ⇒ 不限制策略，只改变"什么算好终点"。
- **`K_SUSTAIN = 25`**：一次抓稳要持续 25 步（0.5 s = `dwell_steps`）才算"推进了一根杆"。它把"留下"的压力和自我课程**耦合**起来——门限越长，goal 越会在"刚抓到的那根杆"上多停一会儿，`hold_next` 才有时间涨上去（25 步 → 0.92，`success` 才可能达到）；抓到 3 步就把 goal 放走的话，`hold_next` 最多 0.26，退化解又会回来。
- 两个"到下一根杆的距离"（米）代替了原来的软接触量（§9.38.4）：**距离在到达之前就单调下降**，是 run #12 缺的"方向"；接触量只在到达瞬间跳变，起不到引导作用。

**CPU 校验（全部通过）**：

| 检查 | 结果 |
|---|---|
| 布局 | `state=148 / goal=6 / obs=154`，`goal_indices=(141..146)` |
| goal_set | 只有一行 = `[−0.071, 0, **0, 0**, 1, 0]`（与杆号无关；`progress=0` = "到达下一根杆并留下"，见 §9.37.3）|
| 平移不变 | 起点随机在任意杆时初始 `dist` 相同（1.767）✓，且 `k_ref = k_start` = 实际起始杆 ✓ |
| **到位且留下** | 挪到 B1 悬挂位姿后：`dist` 0.905(1 步) → 0.424(11 步) → **0.336/0.323(21 步) = success** ✓（`dwell_success` 也因此可达）|
| **推进后重定向** | 持续抓稳 **25 步**（`K_SUSTAIN`）后 `k_ref: 0→1`、`next_streak` 归零、goal 指向 B2、`dist` 回到 1.76 ✓（连续过多杆的自我课程）|
| 事件级 info 复位 | 用 `brax_ext.wrap` 强制掉落：`max_bar/k_ref/cov_b1_runmax` 均回到该 episode 的 t=0 值，`cov_min_d_b1` 回到 2.0 ✓ |

### 9.35.4 顺带修掉的第二个指标 bug：`info` 里的计数器从不跨 episode 复位

brax 的 `AutoResetWrapper` 只换 `pipeline_state`/`obs`，**不碰 `info`** ⇒ 我们放在 `info` 里的所有计数器（`max_bar`、`dwell`、`hold`/`k_ref`、覆盖率 tracker）在掉落重生后**继续累加**。证据：run #11 的 `cov_b1_runmax`(全窗口最长双持段) 一度 **大于** `cov_both_on_steps`(同窗口双持步数)——这在数学上不可能。

修法：env 新增 `episode_info_zero()`（t=0 值，`reset()` 与包装器共用），`MjxAutoResetWrapper` 在 `done` 的环境上按 per-env mask 复位这些键。现在 `max_bar`/`dwell`/`dwell_success`/`kref_max` 都是**真正的 per-episode** 量。`src/check_metrics.py` 也加了第四道检查：`step()` 写入的 metrics 键必须在 `reset()` 里零初始化（`kref_max` 就是被这条抓出来的——它第一次跑就被 `EpisodeWrapper` 的 scan 结构检查拒绝）。

### 9.35.5 run #12 命令（M4.0，1.5 h）

```bash
.venv-warp/bin/python src/check_args.py && .venv-warp/bin/python src/check_metrics.py

.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant advance --start-bar-max 1 \
  --expl-hold 10 --num-evals 20 --steps 12200000 \
  --checkpoint-dir runs/ckpt_adv --save-every 5 \
  --impl warp --scene full035 --wandb --exp-name brach_adv_b1
```

（`--goal-variant advance` 是**单指令目标**，`--train-goal-bar/--eval-goal-bar` 被忽略，同 cross 家族；`--start-bar-max 1` = 从 B0 或 B1 出发各半。）

| 现象 | 判读 | 下一步 |
|---|---|---|
| `advance_max` 停在 0、`len` 掉回 40–70 | 目标偏严（要两只手同时 + 持续 0.5 s），策略退回鱼跃 | 先加 `--entropy-param 1.6` 降噪；或把 `K_SUSTAIN` 降到 10 |
| `advance_max` = 1 出现、`len` ≥ 200 | 能推进一根杆且稳定（M4.0 达成） | 跑满 4 h 看 $P(\ge 2)$ |
| `advance_max` ≥ 2 | 已经连续过多杆 | 直接进 M4.3（9 根杆 + lap 统计） |

## 9.36 复盘：run #11 的成功有多少归功于 `hold`？（以及它为什么必须进 M4 的 goal）

用户问：M4.0 里有没有 $1-e^{-\text{streak}/10}$？它是不是 goal 的一部分？run #11 的成功是否主要靠它？——**前两个答案：M4.0 最初的版本没有，现已加入（`hold_next`，§9.35.3）；第三个答案：是，而且是决定性的。**

**A/B 证据（同一几何，只差这一维）**：

| | run #10 `support` | run #11 `support_hold` |
|---|---|---|
| 几何 | 0.35 m | 0.35 m（相同）|
| goal | `[x,z,p_L,p_R,max(c)]` | 同左 **+ `1−e^{−streak/10}`** |
| len | 41 | **368** |
| fell | 100% | **31%** |
| 最近距离 | 0.006 | 0.005（同一档）|
| `max_bar` 和 | 12 | 289 |

run #10 已经把"够到"解决（0.006 m）却把 `len` 从 67 掉到 41——**"碰一下就走"**；run #11 只加了 `hold` 这一维，`len` 就涨到 368、`fell` 掉到 31%。⇒ **"从擦到变成抓住"这一跃主要由 `hold` 提供。**

机制：`hold` 是 **goal 的一维**（不是 reward），它把"两手都离开杆"的状态在目标空间里推远 1.0 ⇒ run #10 里那个 `dist=0.06` 的"鱼跃最佳瞬间"在 run #11 里变成 `dist≈1.0`，actor 因此被推离"飞掠"。

**但它不是万能的**：run #11 的接触项是 `max(c_L,c_R)`、`hold` 又是"**至少一只手**在窗内" ⇒ 单手吊住就双双满分，第二只手不再被要求（§9.35.2）。它解决的是"**别在无支撑状态下结束**"，没解决"**两只手都要在**"。

**结论（M4.0 的设计依据）**：
1. 只要 goal 里的接触量是**瞬时**的，"飞掠"就会回来 ⇒ M4 必须在 goal 里保留一个**持续量** → `hold_next`（针对**下一根杆**，不是"任意杆"）。
2. "两只手都要在"必须由**每只手各自的、针对目标杆**的接触项表达（`max` 会抹平）→ `c_{L,B_{k+1}}` 与 `c_{R,B_{k+1}}` 分开。
3. 合起来就是 `advance` 的 6 维 goal：**"到了下一根杆、两只手都在、并且留下"**；策略侧仍完全不限制（鱼跃、单手摆荡都允许）。

## 9.37 关于 M4 目标的两点澄清（用户提问）——附带两个 bug 修复

### 9.37.1 "goal 总是初始状态的下一根杆，还是可能隔几根？"

**都不是。** goal 永远指向 **`k_ref + 1`**，而 `k_ref` **在 episode 内自己往前走**：

| 时刻 | `k_ref` | goal 指向 |
|---|---|---|
| reset | 起始杆 $k_0$（`--start-bar-max` 随机）| $k_0+1$ |
| 在 $k_0+1$ 上持续抓稳 25 步（`K_SUSTAIN`）| $k_0+1$ | $k_0+2$ |
| 再一根 | $k_0+2$ | $k_0+3$ |
| 到最后一根（B4）| 4 | 4（被 clamp）= "停在最后一根" |

所以它不是"从初始状态数一根"，而是"**从你上一次站稳的那根数一根**"。**能跳杆**：`k_ref` 的规则是"任何**比当前 `k_ref` 更靠前**且持续抓稳 25 步的杆号" ⇒ 一次鱼跃直接抓稳 $k_{\rm ref}+3$ 会让 `k_ref` 跳过去（跳过两根，`progress` 一次 +3）。但 **goal 从不指超过一根**。

### 9.37.2 "要跑完所有杆，`1−e^{−next_streak/10}` 是否不合适？或只在最后一根加？"

**每一根都需要，而且不能只放最后一根**：它不是终点条件，而是"**每一次到达都必须是真的留下，而不是擦一下**"（run #10→#11 的 A/B，§9.36）。最后一根不需要特殊处理——`k = min(k_ref+1, n_bars−1)` 的 clamp 会让它自然变成"停在最后一根"。

### 9.37.3 但"走完所有杆"确实需要**额外一维**：`progress`（本轮新加）

完全平移不变的目标有个隐患：**"稳定吊在 $k$ 号杆（看向 $k+1$）"和"推进一级之后的稳定状态"在相对量上逐位相同** ⇒ 轨迹大部分时间在稳定悬挂，重标记出的大多数 goal 被当前状态**平凡满足**（对比任务退化，§9.31.1 的坑），没有东西推着策略继续走。

**修法**：goal 加一维 `progress = k_ref − k_start`（原始计数，1 根 = 1.0）。这样"更靠后的未来状态"在 goal 空间里**明确更远**，而到达它唯一的路就是真的再推进一根杆；它仍然是**相对增量**，单指令目标的形式没被破坏。

**代价（已知并记录）**：instruction goal 的 `progress` 置 **0**（"到达下一根杆并留下"）⇒ `eval/episode_success` 只衡量"每次到达是否稳住"，不衡量走了几根。**M4 的头条量是 `advance_max`**（本 episode 推进的杆数，相对起始杆归一，起点在 B1 不会白得 1 分）。

### 9.37.4 顺带修掉的两个 bug（都是 `--start-bar-max` 引出来的）

1. **`k_ref`/`k_start` 的起点**：原来 reset 后恒为 0，而 `--start-bar-max 1` 让一半环境从 B1 出发 ⇒ 第一个 goal 会指向"它正吊着的那根杆"（甚至身后）。现在 reset 把它们设成**实际起始杆**；新增 `k_start` 供 `progress` 用 ✓（实测：从 B1 出发时 `k_ref=k_start=1`，初始 `dist` 与从 B0 出发时相同 = 1.767）。
2. **auto-reset 恢复的是"硬零"而不是"该 episode 自己的 t=0 值"**：`MjxAutoResetWrapper` 现在 reset 时存下 episode 级初始 info（`_first_episode_info`），掉落重生时恢复**那些值**——否则从 B1 出发的环境一生就掉回 `k_ref=0`，bug 1 会在每次重生后复发 ✓（实测：掉落重生后 `k_start/k_ref` 仍与各自起始杆一致）。

**指标改名**：`kref_max` → **`advance_max`**（推进的杆数，相对起始杆）。

## 9.38 run #12（`advance`）失败分析 —— **不是 margin 的锅，是"平移不变目标"把方向信号也消掉了**

### 9.38.1 实测（`runs/brach_adv_b1`，git `d0101b8`，20 evals / 12.2 M 步）

| steps | len | fell | **advance_max** | C(0.1) | 最近距离 | 离杆最长 | B1 双持 | 双支撑 | succ | dwell | dist/len | acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.74 M | 501 | 0% | **0** | 0.44 | 0.164 | 3.1 | 4.2 | 9.8 | 0 | 0 | 1.76 | 0.04 |
| 3.15 M | 501 | 0% | **0** | 0.31 | 0.177 | 4.4 | 3.0 | 23.0 | 0 | 0 | 1.78 | 0.09 |
| 5.56 M | 501 | 0% | **0** | 0.69 | 0.083 | 19.3 | **79.2** | 244.7 | 0 | 0 | 1.77 | 0.07 |
| 9.18 M | 362 | 75% | **0** | 0.69 | 0.052 | 43.4 | 1.7 | 4.8 | 0 | 0 | 1.78 | 0.07 |
| 12.20 M | 388 | 56% | **0** | 1.00 | 0.006 | 33.3 | 0.9 | 8.7 | 0 | 0 | 1.73 | 0.08 |

**`advance_max` 全程 0**（一次"持续抓稳下一根杆"都没发生）；`succ`/`dwell` 全 0；`dist/len` 恒等于**初始距离** 1.77（说明它**根本没往目标靠近**，不是"靠近了但没抓住"）；前 8 个 eval 是典型的"稳定悬挂不动"（len 501、fell 0），之后开始掉落（air 最长 **216 步** = 4.3 s 自由落体）。

### 9.38.2 为什么不是 margin

`margin` = 12 个臂/腰关节的力矩余量，是**有界状态函数**（0–1），悬挂时 0.854、单手伸出（M3.0 的 IK 姿态）0.784 ⇒ 它是**真会随姿态变化**的信号，量级与其它维度相当，没有把距离尺度带偏。它的作用是"什么样的终点算好终点"（留余量），**不会提供/抹掉前进方向**。若真是它，症状应该是"能推进但姿势很费电"，而不是"一步都不动、`dist` 恒等于初始值"。

### 9.38.3 真正的原因：完全平移不变 ⇒ goal 沿"前进方向"是常数

`advance` 原来的 7 维里，去掉 `progress`/`margin` 后只剩 `[dx, dz, c_L,next, c_R,next, hold_next]`。对一个**稳定吊在起始杆**的机器人：`dx` 恒定（≈−0.42 + 摆动）、`dz`≈0、两个接触量恒为 0、`hold_next` 恒为 0、`progress` 恒为 0 ⇒ **当前状态与它自己未来帧的特征几乎逐位相同** ⇒ 重标记出的 goal 被**平凡满足**：

* `categorical_accuracy` 掉到 **0.04–0.10**（成功 run 是 0.34–0.47）⇒ critic 学不到可用的度量；
* 没有**方向**：绝对坐标版本里"目标在 +0.35 m 处"会给 actor 一个明确的推力；平移不变版本把这个系统性偏差**正好消掉**了（这正是它的设计目的），于是"往哪走"完全靠纯探索撞。

这正是 §9.31.1 的教训换了个形式：**goal 必须含有一个连续、不饱和、且"随进展而变"的分量**。`hold_next`/接触量只在**到达之后**才变（0→1 的跳变），到达之前没有梯度。

### 9.38.4 修法：把两个饱和的接触量换成**到下一根杆的距离**（本轮已改）

`advance` 的第 3/4 维从 `c_{h,next}`（饱和，到达前恒 0）改成 **`d_{h,next}`（每只手到下一根杆的距离，米）**，量纲与位置项一致、随伸手单调下降、且仍然平移不变（杆号无关）✓。实测（原生 MuJoCo，`full035`，k_ref=0，`goal_set = [−0.071, 0, 0, 0, 1, 0, 0.854]`）：

| 状态 | 特征 `[dx, dz, d_Lnext, d_Rnext, hold_next, progress, margin]` | 到 goal 的距离 |
|---|---|---|
| 吊在起始杆 | [−0.421, 0, **0.350**, **0.350**, 0, 0, 0.854] | **1.169** |
| **静态伸手：右手搭到下一根杆**（M3.0 的 IK 姿态，左手仍在旧杆，不需换手！）| [−0.421, 0, 0.350, **0.001**, **1.0**, 0, 0.784] | **0.500** |
| 换手到下一根杆（hold 还没建起来）| [−0.071, 0, 0, 0, 0, 0, 0.854] | 1.000 |
| 换手 + 持续 25 步（真实 env 里 hold_next→0.92）| … | **≈0.08** |

⇒ 现在有一条**密集、分级、且部分可达**的课程：**伸手（1.17→0.50，一个静态姿态就能做到，M3.0 已证明 0.7 mm 可达）→ 抓住并保持（→0.08）→ `k_ref` 推进、`progress`+1、goal 自动指向再下一根杆**。这正是 run #12 缺的"第一步"。

### 9.38.5 同轮修掉的崩溃（用户渲染时遇到）

`reset()` 里 `k0` 只在 `if self.start_bar_max > 0:` 分支内赋值 ⇒ **任何 `--goal-variant advance` 而不带 `--start-bar-max` 的调用**（渲染器、`check_reset`、探针）都在 reset 时 `UnboundLocalError`。训练没受影响（它带了 `--start-bar-max 1`），所以是"跑完了但渲染挂了"。已改成先给 `k0 = 0` 再按需覆盖 ✓。

**run #13 命令**（与 #12 完全相同，代码已修）：

```bash
.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant advance --start-bar-max 1 \
  --expl-hold 10 --num-evals 20 --steps 12200000 \
  --checkpoint-dir runs/ckpt_adv2 --save-every 5 \
  --impl warp --scene full035 --wandb --exp-name brach_adv2
```

判据：先看 **`eval/episode_advance_max`**（≥1 = 真推进），其次 `cov_b1_runmax`（双手在下一根杆上的持续段）、`cov_air_runmax`（应明显小于 run #12 的 33–216）、`dist/len`（应**从 1.17 往下走**，run #12 恒 1.77）。若 `advance_max` 仍为 0 但 `dist/len` 明显下降 ⇒ 说明它在"伸手"这半程上学会了，下一步就该上 **M3.4 近目标 reset**（起始就让一只手搭在下一根杆上）把"抓住"这一段喂进去。

## 9.39 M4 goal 设计复盘（十条原则 + 当前方案 + 备选）

### 9.39.1 证据：哪一版 goal 行、哪一版不行、为什么

| goal | 结果 | 死因 / 生因 |
|---|---|---|
| `position`/`full`/`support`（**绝对**坐标 + 瞬时接触）| 学会够杆（最近 6 mm），但"擦一下就走" | 生因：8 个**连续**绝对位置给了 critic 可学的东西 + "目标在 +0.35 m 处"给了**方向**；死因：瞬时接触 + 绝对构型奖励"飞掠" |
| `support_hold`（+ 持续接触，`max(c_L,c_R)`）| run #11：len 41→368、单手吊住 350/400 步 | 生因：`hold` 把"无支撑"的状态推远 1.0 ⇒ 必须**真的抓住**；死因：`max` + "≥1 只手" ⇒ 单手即满分，第二只手不被要求 |
| `cross3`（距离 + 两个饱和接触，3 维）| 崩（acc 0.08）| 只有一个 ~0.11 m 动态范围的连续分量 + 两个饱和量 ⇒ 对比任务退化 |
| `advance` v1（完全平移不变 + 瞬时接触 + hold + progress + margin）| run #12：`advance_max` 恒 0、`dist/len` 恒等于初值 | **沿前进方向是常数** ⇒ 重标记目标被当前状态平凡满足（acc 0.04–0.10），actor 没有方向可走 |
| `advance` v2（本轮：两个接触 → 两只手到下一根杆的**距离**，去掉 margin）| 待跑（run #13）| 距离**在到达前单调下降** ⇒ 方向回来了；分级实测 1.169 → **0.495**（静态伸手，M3.0 已证可达）→ ≈0.08（抓住并保持）|

### 9.39.2 十条设计原则（每条都有上面的实测支撑）

1. **goal 必须含连续、不饱和、与任务相关的分量**（`cross3` 死因）。
2. **它必须"在终点事件之前"就随进展单调变化**，即给出**方向**。绝对坐标是"意外地"提供了方向（固定目标在 0.35 m 外），完全平移不变会**正好把它消掉**（run #12 死因）。
3. **饱和指示量只适合做终点条件**，且必须与"距离版本"配对，让接近过程有分级信号。
4. **终点条件必须是"持续"的**（`hold`/`hold_next`），否则对比学习会farm"擦一下"（run #10 vs #11）。
5. **语义上属于某个部件的量，必须按部件分别表达**：`max(c_L,c_R)` 把"哪只手"抹平了 ⇒ 第二只手永远不被要求（run #11）。
6. **跨杆时必须能区分"进度层级"**：纯局部（一根杆）相对目标会让"再往前一级的稳定状态"与"当前稳定状态"逐位相同 ⇒ 大多数重标记目标平凡（`progress` 维；等价于老设计里的 G2"杆号 one-hot"，但这里用的是相对增量）。
7. **instruction goal（评估用）与训练目标分布是两个不同的对象**：CRL 训练用的是**策略自己未来帧**的重标记目标 ⇒ ① `success` 不是被优化的东西，只能当探针；② 判断设计好坏要看"沿着策略自己的轨迹，重标记出来的目标长什么样"；③ 终点才可达的目标对训练没问题（重标记自带课程），但对评估读数没用。
8. **goal 的"形状"（维度含义）对所有杆、所有起点都一样**——这才是多杆泛化的来源；但**数值**要能编码进度（= 原则 6）。"与杆号无关" ≠ "对进度盲"。
9. **不要放不携带任务信息的维度**：到终点前恒定的量只会稀释距离、加重退化（`margin`、以及偏恒定的 `dz` 都属于此类；`margin` 已删，`dz` 暂时保留作竖向稳定的弱约束）。
10. **重标记采样结构（同轨迹、严格更晚、∝γ^Δt）本身就是设计的一部分**：目标是策略自己的未来 ⇒ **从初始状态不可达的目标在有人碰到它之前不提供任何信号**。所以"**部分可达的前置**"极有价值——v2 里"静态伸手"就值 1.169→0.495，而且 M3.0 已证明它 0.7 mm 可达；这正是 run #12 缺的"第一步"。

### 9.39.3 当前方案（v2）与状态/目标的划分

- **state（148 维）**：绝对量全留着（qpos/qvel/手点/距离矩阵/软接触/`prev_action`）+ `k_ref`（进度上下文）。critic 因此**可以**学"我在第几根杆"，但 goal 不要求它。
- **goal（6 维）**：`[Δx, Δz, d_{L,n}, d_{R,n}, hold_n, progress]` —— 全部是**相对量**，所以同一个 goal 向量在每一根杆上成立。
- **三件事分别由不同维度负责**：*走近*（两个距离 + Δx/Δz）→ *抓住并留下*（`hold_next`）→ *继续往前走*（`progress` + `k_ref` 自推进）。这就是"分级课程"的骨架。
- **没有路径约束**：鱼跃、单手摆荡都允许；只是终点必须是"在下一根杆上、两只手、且持续 0.5 s"。

### 9.39.4 考虑过但没采用的方案

| 方案 | 为什么不用 |
|---|---|
| 绝对坐标 + 每根杆一个 instruction goal（+`--eval-goal-bar`）| 不 scale 到多杆；且奖励飞掠 |
| 杆号 one-hot 进 goal（老 §4 的 G2）| 是绝对量：换一根杆就要换一套目标值 ⇒ 失去"一个 goal 走遍全场" |
| 纯平移不变 + 瞬时接触（v1）| run #12：方向被消掉 |
| 只用接触（`cross3` 式）| 对比任务退化 |
| `hold` 用"任意杆 + `max`"（run #11）| 第二只手不被要求 |
| `margin`（力矩余量 / 省力）| 用户的意图是"可能产生与鱼跃不同的策略"，但不是 M4 重点 ⇒ 已删（保留在 `def6286`）|
| 一次性"多杆"目标（如"到 k_ref+2 或更远"）| 不需要：`k_ref` 自推进已经把"永远再往前一根"表达出来了，加长目标只会更难 bootstrap |
| 加躯干姿态/倾角维 | 可选（`dz` 偏恒定，倾角信息量更大）；等 v2 跑出来再决定 |

### 9.39.5 评估口径（不要让 `success` 骗人）

| 读数 | 含义 |
|---|---|
| **`advance_max`** | 本 episode 推进的杆数（相对起始杆）——M4 头条 |
| `dist/len` | 平均每步到（当前）目标的距离；run #12 恒 1.77 = 完全没动，v2 应从 **1.17 往下走** |
| `cov_b1_runmax` / `cov_xfer_runmax` | 在下一根杆上双手持续多久 / 是否出现"一手旧杆一手新杆"的换手态（run #10/11 都是 0）|
| `cov_air_runmax` | 滞空时长：run #12 高达 33–216 步（乱飞），run #10/11 ≈ 11–19 步 |
| `success` / `dwell_success` | 只作"单次到达是否稳住"的探针，**不衡量走了几根** |

### 9.39.6 若 run #13 仍卡住，按这个顺序动

1. `dist/len` 降到 ~0.5 但 `advance_max` = 0 ⇒ "伸手"学会了、"抓住"没有 ⇒ **M3.4 近目标 reset**（起始就让一只手搭在下一根杆上）把抓住那段直接喂进 replay。
2. 全部指标都不动 ⇒ 检查探索尺度（`--entropy-param 1.6` / `--expl-hold 20`）。
3. 能推进但不稳 ⇒ 调 `K_SUSTAIN`（门限长度就是"留下"的压力强度）、`n_bars` 扩到 9、`γ` 0.995→0.997。
