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

## 9.40 采用用户 M4 文档的方案：`support_dual`（run #11 只加一维 `h_dual`）

依据：`docs/M4_对与#11的分析和goal的设计问题的讨论.md`（用户撰写）。**结论：同意，并已实现。**

### 9.40.1 我对该文档的评估

**同意的部分（逐条 + 我补的证据）**

| 文档观点 | 我的评估 |
|---|---|
| 不要替换 run #11 的全局 `h_any`，它是被实验验证的机制 | ✓ 同意。`h_any` 把"飞掠"从 `dist≈0.06` 推到 `≈1`，这是 run #10→#11 的主因 |
| 只加**一维** `h_dual = 1-e^{-S_dual/10}`，`S_dual` = "两只手同时在同一根杆 5 cm 窗内"的连续步数 | ✓ 已实现（`support_dual`）。这是修"单手侧吸引子"的**最小改动** |
| 不要上"每只手×每根杆"的 10 维 streak | ✓ 同意：与绝对 `p_L,p_R` 重复，且维度随 $n_{\rm bars}$ 增长 |
| 不要用 `min(c_L,c_R)` | ✓ 同意：仍然瞬时；run #11 ep0 step21 就已经 `c_L=c_R=1`，加它等于重演 run #10 |
| 两条 streak 都用 **5 cm** 判据（不是 `max(c)>0.5` 的 3.3 cm） | ✓ 同意（实测稳定悬挂会漂到 3–4.5 cm，3.3 cm 会让 streak 闪烁归零）|
| 第一枪 `/10` 不要改，保持"只加一维"的干净 A/B | ✓ 同意（run #13 = run #11 + `h_dual`）|
| `h_any`/`h_dual` 也必须进 **state**（否则是 POMDP，会污染 critic）| ✓ 同意，**而且代码里已经如此**：goal 必须是 state 的切片，所以 `support_hold` 的 `h_any` 本来就在 state 里（state 143 / obs 153）；新变体把 `h_dual` 也放进 state（144 / 155）|
| 指数 `1-e^{-S/τ}` 的价值不止"持续时间"，更是给 hindsight 造了一条**连续可自举的目标轴**（3→5→10→20→25 步 ⇒ 0.26→0.39→0.63→0.86→0.92）| ✓ 强烈同意，而且**正样本比文档估计的还多**（见下）|
| 加 `dual_*` 指标来检验"侧吸引子"假设 | ✓ 已实现 4 条 |

**用回放数据加强文档的一点**：文档说"只观察到 3 步双持"。实际上 run #11 最终策略的**确定性回放**（`runs/render/hold_final/`，3 集）里，第 1 集的**双手同挂 B1 连续 52 步**（另两集 2 步和 7 步）。也就是说"双手稳定支撑"这个行为**已经真实出现过**，`h_dual` 能拿到的正样本上限不是 0.26 而是 **0.994**（$1-e^{-52/10}$）。⇒ 文档"不是要求策略发明全新行为，而是给已经偶然出现的行为加一个可被捕获的坐标"这个论证，比文档自己写的更强。

**我要补的两点（文档没覆盖）**

1. **`c_sup = max(c_L,c_R)` 这一维的标定问题（既有、且会影响 run #13 的读数）**：它的 goal 值来自关键帧的 `c=1`（手"埋"在指笼里），但**稳定悬挂时抓手点会漂到 3–4.5 cm ⇒ 实测 `max(c)≈0.55`**，于是 $(1-0.55)^2≈0.20$ 是每个距离里一个**下不去的常数**。实测：吊在起始杆时 `dist=0.76`（= $\sqrt{0.606^2+0.20+0.005}$ ✓ 完全吻合）。后果：**即使完美到位 + 双持 25 步，`dist` 也只能到 ≈0.45 > 0.35 ⇒ `success` 对"稳定悬挂"永远不亮**（run #11 里 `succ` 只在撞击/受力把手"埋"深时才亮，正是这个原因）。
   ⇒ run #13 仍按文档保持干净 A/B（不动 `c_sup`），但**读数请以 `dual_*`、`len`、`fell`、`dist` 为准，别用 `success`**；下一枪（run #14）建议把 `c_sup` 从 goal 里去掉（它现在与 `h_any`/`h_dual` 语义重复，且目标值不可达）。
2. **命名/编号**：文档里的"run #12"在我们日志里是 run #13（run #12 = 我那条 `advance` 平移不变目标线，已停在代码里、未采用）。这次按文档走 `support_dual`。

### 9.40.2 实现（`--goal-variant support_dual`）

$$g=\big[\underbrace{x,z,p_L(3),p_R(3)}_{\text{绝对位置（8）}},\;\underbrace{\max(c_L,c_R)}_{\text{接触}},\;\underbrace{h_{\rm any}}_{1-e^{-S_{\rm any}/10}},\;\underbrace{h_{\rm dual}}_{1-e^{-S_{\rm dual}/10}}\big]\quad(11\ \text{维})$$

- `S_dual(t+1) = S_dual(t)+1` 若"**存在一根杆，两只手都在它 5 cm 窗内**"，否则归零；`S_any` 沿用 run #11 的定义（至少一只手在任意杆窗内）。
- state = 144（141 + `c_sup` + `h_any` + `h_dual`），goal = 11，obs = 155；`goal_indices = (0,2,99..104,127,128,129)`。
- **两个 streak 都在 state 里**（文档要求；goal 是 state 的切片，所以这是自然结果）。
- 新指标（评估器里已加）：
  | 指标 | 求和 = |
  |---|---|
  | `cov_dual_runmax_improve` | 本集 **最长** $S_{\rm dual}$（文档的 `dual_streak_max`）|
  | `cov_dual_on_steps` | 双持步数（文档的 `dual_support_steps`）|
  | `cov_single_on_steps` | 单手支撑步数（文档的 `single_support_steps`）|
  | `cov_dual_hold_sum` | $\sum_t h_{\rm dual}(t)$（双持质量的时间积分）|
- **三层结构数值验证**（CPU，HOLD/松手三种状态）：

  | 状态 | `h_any` | `h_dual` | `S_dual` | dist |
  |---|---|---|---|---|
  | 双手挂起始杆（dual）| 0.95 | **0.95** | 30 | 0.76 |
  | 左手松开（single）| 0.99 | **0.00** | 0 | 1.25 |
  | 双手都松开（flight）| **0.00** | 0.00 | 0 | 4.03 |

  ⇒ goal 空间里 flight → single → dual 三层被明确区分（single 比 dual 远 0.5），正是文档要的几何。
- ⚠️ 与 run #11 的 checkpoint **不兼容**（obs 153→155）⇒ 新跑。

### 9.40.3 run #13 命令与判据

```bash
.venv-warp/bin/python src/check_args.py && .venv-warp/bin/python src/check_metrics.py

.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant support_dual --train-goal-bar 1 --eval-goal-bar 1 \
  --expl-hold 10 --num-evals 20 --steps 12200000 \
  --checkpoint-dir runs/ckpt_dual --save-every 5 \
  --impl warp --scene full035 --wandb --exp-name brach_dual_b1
```

（与 run #11 逐项相同，**只把 `--goal-variant support` 换成 `support_dual`** ⇒ 干净的 +1 维 A/B。）

| 曲线 | 预期（若"侧吸引子"假设正确）| 若相反 |
|---|---|---|
| `cov_dual_runmax_improve` | 3 → 5 → 10 → 20 …（**先于** episode 变长上升）| 长期卡在 2–3 |
| `cov_dual_hold_sum` / `cov_dual_on_steps` | 单调上升 | 不动 |
| `cov_single_on_steps` | 先升（探索期）后**下降** | 一直高 |
| `len` / `fell` | 在 dual 指标之后跟着改善 | 不动 |
| `dist`（**别看 `success`**，见 9.40.1）| 从 ~0.78 往下 | 不动 |

若 $S_{\rm dual}$ 长期卡 2–3 且右手每次到杆后**物理滑脱** ⇒ 按文档的判据：问题已不在 goal，而在**抓握动力学 / action representation / 接触控制**，那时再转向那条线（M3.0 的接触/抓手分析、指力、`expl_hold` 尺度）。

## 9.41 run #13 复盘：指标说"双持 46%"，视频却是"单手挂 B1 + 另一只手回够 B0"

### 9.41.1 事实（`runs/render/dual_final/`，3 集 × 401 步，确定性策略）

每只手的"最近杆号"分布（步数）：

| 集 | 左手 | 右手 | 双持所在杆 |
|---|---|---|---|
| 0 | B1: **337**（离杆 58, B0 6）| B1: **162**（离杆 231, **B0 8**）| B1 162 / B0 6 |
| 1 | B1: **322**（离杆 73, B0 6）| B1: **78**（离杆 314, **B0 9**）| B1 78 / B0 6 |
| 2 | B1: **350**（离杆 45, B0 6）| B1: **147**（离杆 245, **B0 9**）| B1 147 / B0 6 |

时间历程（ep1）：左手从换手完成后一直挂 B1；右手大部分时间是 `-1`（离杆），`dist` 常驻 1.1–1.7；只在 step 320 与 400 才回到 B1（`dist` 0.27 / 0.095）。

⇒ **"单手挂 B1、另一只手偶尔真的回到 B0"是真实行为**（每集都有 8–9 步 `bar_R=0`），不是渲染问题。

### 9.41.2 为什么 eval 指标看起来不错（三条，都不矛盾）

1. `cov_dual_*` 是 **16 个 eval env 的逐步平均**；这 3 集恰好偏弱（`cov_dual_on_steps`：eval 219.9 vs 回放 162/78/147，`dist/len`：0.83 vs 1.2–1.7）。
2. **双持"成段但不维持"**：最长段 53–62 步（eval max 69.8 ✓），基本发生在**刚落上 B1 之后**；随后右手松开摆回，隔一阵再搭回来。所以 `dual_on_steps/len = 46%` 只能读作"**存在双持段**"，不能读作"**一直双持**"。
3. 指标测的确实是 B1（`dual_on_bar` 绝大部分是 1，不是 0）⇒ 指标没坏，是**比例给人的印象**有偏差。

### 9.41.3 机制：这是"过渡进行到一半"，不是学不会

- goal **确实在推**右手回来：单手 `dist≈1.2` vs 双手 `≈0.1`（差 ~1.1，主要来自 `h_dual`）。
- 但 CRL 训练目标是**策略自己未来帧**的重标记目标；当前多数未来帧是"单手挂 B1 + 右手乱摆" ⇒ goal 分布多数仍要求该姿态 ⇒ run #11 的侧吸引子**被削弱但没消失**。
- 正向反馈**正在起效且没有平台**：`S_dual` 5→24→37→56.5→**69.8**；`dual_on_steps` 7.6→**219.9**；`single_on_steps` 493→**209**；`len` 501、`fell` 0–6%。

### 9.41.4 可选的杠杆（按性价比）

1. **让一部分 episode 起手就处于"双手挂目标杆"**（放开 `--start-bar-max` 的"仅 advance"限制 + 目标杆采样）⇒ dual 姿态立刻进入 replay，正反馈提前闭合（M3.4 的 M4 版）。
2. **同配方跑更久**（本次采用 6 h）。
3. **拉开单手/双手差距**：`max(c_L,c_R)` → 每只手各自的 `c_L,c_R`，并加两个**每只手**的 hold（`h_L,h_R`，不含杆号信息，仍不是文档反对的 10 维向量）⇒ 单手要多付 ~2–3 而非 1.0。注意这只改**权重**，不改"目标来自自身未来"的机制。
4. 多杆（goal-bar 采样）押后：**下一跳的发射状态就是"双手稳定挂住"**，现在每集只有 20–40% 时间真正双持。

**读数纠正**：`c_sup = max(c_L,c_R)` 的目标 1.0 在稳定悬挂时不可达（实测 ~0.55），它在单手/双手两种状态下是**同一个 ~0.2 的常数** ⇒ 删它只让 `dist`/`success` 变好看，**不改变单/双手差距**（差距 0.7 全在 `h_dual`）。

### 9.41.5 run #14：同配方 6 h（本次采用）

`--steps` / `--num-evals` 的换算（实测吞吐 8.387 M env-steps/h；prefill = ceil(1000/62)=17 unrolls = 134,912 步）：

$$\text{actual} = \Big\lfloor\tfrac{\text{steps}-128000}{\text{num\_evals}\cdot 128\cdot 62}\Big\rfloor \cdot \text{num\_evals}\cdot 7936 + 134912$$

本仓校准：`12.2M/20evals → 12,197,632 (1.45 h)`、`33.46M/40evals → 33,466,112 (3.99 h)`、**`50.3M/60evals → 50,131,712 (5.98 h)`** ✓

```bash
.venv-warp/bin/python src/check_args.py && .venv-warp/bin/python src/check_metrics.py

.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant support_dual --train-goal-bar 1 --eval-goal-bar 1 \
  --expl-hold 10 --num-evals 60 --steps 50300000 \
  --checkpoint-dir runs/ckpt_dual6h --save-every 5 \
  --impl warp --scene full035 --wandb --exp-name brach_dual_b1_6h
```

**判据（按重要性）**：

| 读数 | 目标 | 说明 |
|---|---|---|
| `cov_dual_on_steps / len` | 从 0.46 → **>0.7** | 真正"一直双持"才算过渡完成 |
| `cov_dual_runmax_improve` | 70 → **150+** | 单次双持时长（3 s）|
| `cov_single_on_steps` | 209 → 继续降 | 与上一条互为镜像 |
| `bar_R` 的"回 B0"现象 | 回放里应消失 | 每集 8–9 步 `bar_R=0` 是当前最刺眼的问题 |
| `len` / `fell` | 保持 474–501 / ≤6% | 别用 `success`（受 `c_sup` 常数拖累）|

若 6 h 后 `cov_dual_on_steps/len` 仍 <0.7 且 `bar_R` 仍回 B0 ⇒ 上 9.41.4 的第 1 条（起手双持 reset），再不行才动第 3 条（每手接触+每手 hold）。

## 9.42 `--start-bar-max` + `--goal-ahead`：真正实现"初始在前 N−1 根任意一根、目标在它后面的任意一根"

**澄清（用户提问）**：`--train-goal-bar -1 --train-goal-bar-min 0` **没有**这个语义——start 恒为 B0（`--start-bar-max` 默认 0），而 goal 是 `gk ~ U[goal_bar_min, n_bars)` 的**独立**采样，与起始杆无关（所以 1/5 的 episode 目标就是它正踩着的那根；一旦放开 start 还可能采到身后）。而且 `--start-bar-max` 当时只允许 `advance`，配 `support_dual` 会直接报错。

**本轮改动**：

1. 放开限制：`start_bar_max > 0` 现在对**任何 goal_set 按杆给出的变体**都合法（目标坐标是绝对量，无论从哪根杆出发都正确）；唯一不允许的组合是"随机起点 + 固定目标杆"（会把目标放到身后）。
2. 新增 `--goal-ahead 1`：`gk = min(k0 + 1 + U{0, …, n−1−k0}, n−1)` ⇒ **目标恒在起始杆之后**（要求目标杆是采样的，即 `--train-goal-bar -1`）。
3. **CPU 验证**：`start_bar_max=3, goal_ahead=1`，12 个环境 → `k0 ∈ {0,1,2,3}`、`gk ∈ {1..4}`、**全部 `gk > k0`** ✓

**命令（真正的多杆设定）**：

```bash
.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant support_dual \
  --train-goal-bar -1 --goal-ahead 1 --start-bar-max 3 --eval-goal-bar -1 \
  --expl-hold 10 --num-evals 20 --steps 12200000 \
  --checkpoint-dir runs/ckpt_dual_goal --save-every 5 \
  --impl warp --scene full035 --wandb --exp-name brach_dual_goal
```

- `--start-bar-max 3` = 起始杆 ∈ {B0..B3}（前 N−1 = 4 根）；`--goal-ahead 1` = 目标 ∈ {k0+1..B4}；`--train-goal-bar-min` 此时无意义（下界由 k0 决定）。
- `--eval-goal-bar -1`（**不再钉 B1**）：评估环境与训练同分布（随机起点 + 朝前的目标），因此 `success`/`dist` **不与 run #13 可比**，这一枪的头条是 **`advance_max`**（推进杆数）与 `bar_L/bar_R`/`max_bar` 的杆号分布；`cov_dual_*`（双持质量）仍可与 run #13 比。

## 9.43 `--goal-ahead 1` 的含义、合理性，以及一个必须修的采样 bug

**含义**：让**采样到的目标杆严格位于本 episode 起始杆之后**

$$g_k=\min\big(k_0+1+U\{0,\dots,n-2-k_0\},\;n-1\big)$$

- 不加它（`--train-goal-bar -1 --train-goal-bar-min 0`）：目标杆是**独立**均匀采的，可以等于起始杆（平凡达标），放开随机起点后还可能采到**身后**。
- 加了它之后（5 根杆、起点 ∈ {B0..B3}）目标距离的分布：

  | 距离 d=gk−k0 | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|
  | 理论 | 0.521 | 0.271 | 0.146 | 0.062 |
  | 实测（400 环境）| **0.522** | 0.253 | 0.160 | 0.065 |

  平均 **1.75 根之前**（我上一轮口误说"平均 2.5 根"，实测是 1.75）；10 种 (k0,gk) 组合全部出现，`gk > k0` 恒成立 ✓

**为什么合理（不是可选项）**：如果目标**永远只差一根**，最优策略就是"换一次手然后停" ⇒ rollout 不会往更远处走 ⇒ 重标记出的目标永远不会覆盖 B2 以后 ⇒ 学不会多杆。run #13 的 `advance_max ≡ 1` 就是这个现象的直接证据。所以目标分布**必须**包含远处的杆；而 52% 是"恰好一根"，平均难度仍不高 ✓。

**两个约束**：① eval 必须钉住（起点 B0 + 固定目标杆），否则每个 eval 窗口采到不同难度、曲线不可比 —— 已在 `1a48112` 做了；② `start_bar_max ≤ n_bars−2`（起点在最后一根就没有"之后的杆"了，目标会退化成平凡）—— 已加断言。

**⚠️ 实现 bug（用户这一问才查出来，已修）**：第一版里 `k0` 和 `gk` **共用同一个随机 key**（`randint(rng_goal,…)` 两次），同一个 key 的均匀比特同时决定两个抽样 ⇒ 两者强相关。实测 12 个环境里 `d = gk−k0` 有 11 个是 1 ⇒ **"目标可以是后面的任意一根"悄悄退化成"总是恰好下一根"**。修法：`rng_bar, rng_goal_sel = jax.random.split(rng_goal)`，`k0` 用前者、`gk` 用后者；旧路径（`goal_ahead=0` 且 `start_bar_max=0`）**仍用原来的 `rng_goal`**，所以 run #10/#13 等的随机流逐位不变 ✓。修后实测分布与理论完全吻合（上表）。

若无这个修复，那条 6 h 命令实际跑的就近乎"run #13 + 随机起点"，几乎是白跑 —— 所以这一问很有价值。

## 9.44 run #14（随机起点 + 朝前任意距离目标，6 h）**失败**：goal 一旦"够不到"，就退回旧吸引子

**结果**（`runs/brach_dual_goal6h`，60 evals，50.13 M 步，git `656a931`）：

| steps | len | fell | `advance_max` | `dualmax` | `single` | bar_L | bar_R | `max_bar` | dist/len | acc | critic |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.13 M | 501 | 0% | **0** | 2.7 | 467 | −29 | −496 | **0** | 2.91 | 0.585 | 1.72 |
| 25.97 M | 501 | 0% | **0** | 3.4 | 479 | −19 | −498 | **0** | 2.67 | 0.386 | 2.40 |
| 46.80 M | 47 | 100% | **0** | 1.9 | 9.6 | −36 | −45 | **0** | 3.27 | 0.443 | 2.20 |
| 50.13 M | 16 | 100% | **0** | 3.0 | 6.6 | −7 | −13 | **0** | 2.80 | 0.439 | 2.26 |

**失败特征**：`advance_max` 全程 0（连**第一次换手**都没做，`max_bar` 恒 0）；`dualmax` ≤ 4（双持能力**丢失**，run #13 是 70）；行为在两个旧吸引子之间来回——**单手稳定悬挂**（`single`≈477、`len`=501、`fell`=0，即 run #11 的侧吸引子）或**立刻掉落**（`len`=16、`fell`=100%）。而 critic 是**历次最健康**的（`acc` 0.36–0.585、`critic_loss` 1.7–2.7）⇒ **不是表征/优化崩，而是"目标作为优化对象"失效了**。

**原因**：instruction goal 变成**不可达**。随机起点（k0∈{0..3}）+ 朝前**任意距离**（平均 1.75 根、48% ≥2 根、最远 4 根）⇒ 从起点看目标距离常驻 2.4–2.9（run #13 只有 0.6）⇒ **移动不划算**（任何动作都可能掉，而不动至少"距离不变"）⇒ rollout 退回两个局部吸引子 ⇒ 由这些轨迹重标记出的 goal 里**没有任何"走到远处的成功状态"** ⇒ 自我课程退化。这与 run #12 是**同一类**失败：目标必须是"可达的、带方向的势函数"。

**方法学教训（重要）**：这一枪把**两件事混在了一起**——"起点随机化"和"目标变远"。run #13 证明"目标只差一根 + 起点 B0"能学；把距离放大到 1.75–4 根就毁掉了已有技能。所以下一步必须**拆开**：

| 方案 | 内容 | 回答什么问题 |
|---|---|---|
| **A**（本轮已实现，推荐先跑）| 起点随机（`--start-bar-max 3`）+ 目标**恰好一根**（新增 `--goal-ahead 1 --goal-ahead-max 1`，实测 200 环境 `gk = k0+1` 恒成立 ✓）| 换手技能能否**跨杆泛化**、以及**连续过杆**是否会从重标记目标里自行长出来（看 `advance_max` 台阶）|
| B | run #13 原配方跑 6 h（起点 B0、目标 B1）| 单手→双手过渡能否自己走完（我们原本要的那个数）|
| C | `advance`（平移不变 + 到下一根杆的距离 + `hold_next` + `progress`）| **自推进目标**：instruction goal 永远是"当前站稳的下一根"（距离恒 0.6–1.2，始终可达），多杆由 `progress`/`k_ref` 推进。它的唯一已知阻塞（缺方向）已在 `bf34058` 修好，但**从未重跑** |

**A 的命令（1.5 h 先探）**：

```bash
.venv-warp/bin/python src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant support_dual \
  --train-goal-bar -1 --goal-ahead 1 --goal-ahead-max 1 --start-bar-max 3 --eval-goal-bar 1 \
  --expl-hold 10 --num-evals 20 --steps 12200000 \
  --checkpoint-dir runs/ckpt_dual_next --save-every 5 \
  --impl warp --scene full035 --wandb --exp-name brach_dual_next
```

（`--eval-goal-bar 1`：eval 固定起点 B0、目标 B1 ⇒ 与 run #13 直接可比；多杆看 `advance_max`/`bar_*`。跑 6 h 就把 `--steps/--num-evals` 换成 `50300000/60`。）

## 9.45 run #15（随机起点 + 目标恰好一根，1.5 h）**同样没成功，但失败原因不同：预算被摊薄**

`runs/brach_dual_next`（20 evals，12.2 M 步，git `b153a81`）

| steps | len | fell | `advance_max` | `dualmax` | `single` | bar_L | bar_R | dist/len | acc |
|---|---|---|---|---|---|---|---|---|---|
| 0.74 M | 293 | 100% | 0 | 21.2 | 132 | −245 | −156 | 1.50 | 0.42 |
| 4.36 M | 501 | 0% | 0 | 13.4 | 444 | −445 | −25 | 1.28 | 0.51 |
| 12.20 M | 84 | 100% | 0 | 2.9 | 48 | −79 | −31 | 1.65 | 0.44 |

**行为**（eval 固定起点 B0、目标 B1）：4.36 M 那一窗 `bar_R`=−25 ⇒ 右手在 **B0 上挂了 476/501 步**，`bar_L`=−445 ⇒ 左手只搭了 56 步，`dualstep`=44 ⇒ **又是一手挂住不动**（run #11 的侧吸引子），`max_bar`=0、`advance_max`=0 ⇒ **连第一次换手都没做**。critic 依旧健康（acc 0.43–0.56、critic_loss 1.8–3.0）。

**诊断：这不是新的失败模式，而是 run #13 的"早期阶段"，只是每任务数据少了 4 倍。**
run #13（单一任务：起点 B0、目标 B1）**第一次出现换手是在 6.17 M 步**（≈ 它 12.2 M 预算的一半），而 run #13 在 0.74–4.36 M 那一档的表现正是"一手挂住不动"（`dualmax` 2.5–3.4、`len`=501、`single` 主导）——与本轮同档位几乎一样。本轮把 12.2 M 摊到 **4 个 (起点,目标) 组合**上 ⇒ 每任务 ~3 M ⇒ **低于 run #13 出现换手所需的阈值** ⇒ 卡在早期是**预期结果**，不能据此否定"随机起点 + 目标恰好一根"这个设计。

**下一步（按性价比）**：

| 方案 | 内容 | 说明 |
|---|---|---|
| **A′（推荐）** | 同命令跑 **6 h**（50.13 M ⇒ 每任务 ~12.5 M，与 run #13 相当）| 直接检验"摊薄"解释：若 6 h 后 eval `advance_max`≥1 且 `dualmax` 上升 ⇒ 成立 |
| B′ | 降多样性：`--start-bar-max 1`（2 个任务，每任务 ~6 M @1.5 h）| 最快的 1.5 h 探针 |
| C | `advance`（自推进目标）：instruction goal 永远是"当前站稳的下一根"⇒ **任务只有一种**（杆号只是平移）⇒ 结构上不存在摊薄；其唯一已知阻塞（缺方向项）已在 `bf34058` 修好但未重跑 | 架构上最适合 M4，风险是相对目标的老问题 |
| D | run #13 原配方 6 h（起点 B0、目标 B1）| 已知可行路径，先把双持过渡跑完 |

**A′ 命令**：与 run #15 完全相同，只把 `--num-evals 20 --steps 12200000` 换成 `--num-evals 60 --steps 50300000`、exp/ckpt 改名（`runs/ckpt_dual_next6h`）。

**读数判据**：`eval/episode_advance_max` ≥1（第一次换手出现）；`cov_dual_runmax_improve` 从 ~3 升到 ≥30；`cov_single_on_steps` 与 `len` 同步改善。若 6 h 后仍为 0 且行为停在"一手挂住不动" ⇒ 摊薄解释被否证，那时才转向 C/D。

## 9.46 近几轮实验总览（run #5–#15：设计 × 结果 × 结论）

> 本节把 §9.28–§9.45 串成一条线，方便一眼看清"每轮只改了什么、换来了什么、下一步为什么这么走"。
> 所有数字都是从各 `runs/<exp>/progress.csv` 与 `args.json` 实读（`--seed 0`，单 seed）。跨设备部署的过程不在本节范围内。

### 9.46.1 设计谱系（每轮只改一个变量）

| # | 相对上一轮只改 | 依据（为什么改） |
|---|---|---|
| 5 | （基线）`support` + 场景 `full` + **`m1` 动作窗** | M2 的默认动作参数化 |
| 6 | **动作窗 `m1`→`reach`** | M3.0 静态 IK 证明 `m1` 窗（臂 ±0.2 rad）**物理上够不到 B1**：所需动作值 (2.3, 4.4, 8.1) 被 clip |
| 7 | **`--expl-hold 10`**（噪声时间相干 0.2 s） | 50 Hz 逐步独立噪声在 20 ms 内自相抵消，而摆荡需要 0.1–0.5 s 同向作用 |
| 8 | **goal 语义**：`support` → `cross3`（3 维） | 想让 goal 直接表达"右手到 B1 且左手仍在 B0" |
| 10 | **几何 `0.40`→`0.35 m`**（新资产 `full035`） | M3.0：0.40 时静态极限只有 3.1 cm（判据 5 cm 的 62%），0.35 时 0.0007 m |
| 11 | **goal + 持续接触** `support_hold = support + [1-e^{-streak/10}]` | run #10 学到的是"鱼跃"：飞掠瞬间 `dist≈0.06` 即满足目标 |
| 12 | **平移不变自推进目标** `advance`（相对量 + `k_ref` + `progress`） | 想让"同一个 goal 在每根杆上成立"，从而支持多杆 |
| 13 | **goal + 同杆双持持续维** `support_dual = support_hold + [h_dual]` | run #11 的接触项用 `max(c_L,c_R)`+“≥1 只手” ⇒ **单手吊住就满分**（侧吸引子） |
| 14 | **多杆**：`--start-bar-max 3` + `--goal-ahead 1`（目标朝前任意距离，6 h） | 用户诉求："起点可在前 N−1 根任意一根，目标应是它后面的任意一根" |
| 15 | **同上，但目标恰好一根**（新增 `--goal-ahead-max 1`），1.5 h | 隔离 run #14 里混在一起的两个变量（随机起点 vs 目标变远） |

### 9.46.2 结果总览

| # | exp | len（末） | fell（末） | 最近距离（最好） | C(0.1) | succ（末） | dist/len | acc（末） | 关键新指标 |
|---|---|---|---|---|---|---|---|---|---|
| 5 | `brach_support_b1`(4h) | 414 | 75% | — | — | 0 | 1.05 | 0.43 | `max_bar`**恒 0**：~9000 万步没前进 |
| 6 | `brach_reach_b1_short` | 60 | 100% | — | — | 0.2 | 1.33 | 0.31 | `max_bar`**首次非 0**（4.4）⇒ 动作窗确是被卡的原因 |
| 7 | `brach_reach_b1_a2p` | 68 | 100% | **0.013** | 1.00 | 1.8 | 1.32 | 0.26 | 摆幅 0.067→0.33；**够杆解决、但 100% 掉落** |
| 8 | `brach_cross3_short` | 15 | 100% | 0.236 | 0.00 | 0 | 1.05 | **0.08** | 6 次里最差：3 维饱和 goal ⇒ 对比任务退化 |
| 10 | `brach_d035_reach` | 41 | 100% | **0.006** | 1.00 | 7.1 | 0.67 | 0.34 | `max_bar` 12.1；最好一步 `dist=0.06` ⇒ **"双手离杆鱼跃"** |
| 11 | `brach_hold_b1` | 306 | 44% | **0.003** | 1.00 | 108 | 0.57 | 0.19 | `b1_runmax` 24.5（双持段）、滞空 16 步；回放：左手在 B1 上 349/400 步 |
| 12 | `brach_adv_b1` | 388 | 56% | 0.006 | 1.00 | **0** | **1.73** | **0.08** | `advance_max` **恒 0**；`dist/len` = 初始值 ⇒ **一步没动** |
| 13 | `brach_dual_b1` | **474** | **6%** | **0.002** | 1.00 | **152** | 0.83 | 0.19 | **`dual_runmax` 5→69.8 步（1.4 s）**、`dual_steps` 7.6→219.9、`single` 493→209、`advance_max`=1 |
| 14 | `brach_dual_goal6h`(6h) | 16 | 100% | 0.143 | 0.06 | 0 | 2.80 | 0.44 | `advance_max` 0、`dualmax` ≤6：**目标够不到 ⇒ 退回旧吸引子** |
| 15 | `brach_dual_next` | 84 | 100% | 0.027 | 0.94 | 0 | 1.65 | 0.44 | `dualmax` 峰 29 但末 2.9 ⇒ **停在 run #13 的早期阶段** |
| 17 | `brach_dual_b1_6h`(6h) | **501** | **0%** | **0.000** | 1.00 | **416** | **0.198** | 0.18 | **`dual_runmax` 69.8→424.6 步（8.5 s）**、`bar_R` −0.08→**+0.86**（右手补上 B1）、`hand_on` 95%、`air` 仅 10.6 步 ⇒ 受控换手；**单手侧吸引子解决**（§9.50.4）|
| 18 | `brach_dual_nomax_b1`(1.5h) | 128 | **100%** | 0.018 | 1.00 | **0** | 1.389 | 0.26 | **证伪**：删 `max(c_L,c_R)` ⇒ eval 1 的完美双持被训练拆掉，`advance`/`succ` 恒 0、只掠过 B1 1.8 cm（§9.50.2）|

（`—` = 该 run 早于覆盖率指标（§9.32）加入，没有这些列；`dist/len` = 每步平均到目标的距离，最灵敏的"有没有动"指标。）

### 9.46.3 逐轮结论

1. **#5→#6：动作空间是第一道硬门槛。** 前四次 run 在 `m1` 窗下物理上不可能够到 B1（M3.0 的 IK 解需要 (2.3,4.4,8.1)）；换 `reach` 窗后 `max_bar` 立刻非 0。
2. **#7：够杆解决（最近 1.3 cm、摆幅 0.33 m），但"碰一下就掉"**——goal 里的接触量是**瞬时**的。
3. **#8 否证"只改 goal 语义"**：3 维饱和 goal（`cross3`）让对比学习退化（`acc` 0.08），比 `support` 差得多。
4. **#10：几何修好后策略学成了鱼跃**（`dist=0.06` 的飞掠瞬间即满足目标，`len` 反而 67→41）。
5. **#11：把"持续接触"写进 goal，是"从擦到变成抓住"的决定性一维**（同一几何的干净 A/B：`len` 41→306/368、`fell` 100%→31–44%，最近距离几乎不变）。但它**只解决"别在无支撑状态下结束"**，`max(c_L,c_R)`+“≥1 只手”让**单手吊住即满分**。
6. **#12 否证"平移不变目标"**：goal 沿前进方向是常数 ⇒ 重标记目标被当前状态平凡满足（`acc` 0.08）⇒ 策略一步没动（`dist/len` 恒等于初值）。
7. **#13 成功打断单手侧吸引子**：只加一维 `h_dual`（同杆双手持续）⇒ `S_dual` 5→**69.8 步**、双持占一集 **46%**、`len` 474–501、`fell` 6%、`succ` 152。回放显示 3/3 集整集不掉、双手稳定悬挂 1.2 s。**但**多数时间仍是单手（`dual_on_steps/len`=0.46，"存在双持段"≠"一直双持"），且右手偶尔回够 B0（§9.41）。
8. **#14/#15：多杆的两次失败，原因不同**——#14 是"目标够不到"（平均 1.75 根、最远 4 根 ⇒ 移动不划算 ⇒ 退回旧吸引子）；#15 把目标限制成恰好一根后没有崩，但对 4 个 (起点,目标) 组合而言**预算被摊薄 4 倍**（每任务 ~3 M，低于 run #13 出现换手所需的 ~6 M）⇒ 停在 #13 的早期阶段。

### 9.46.4 当前状态

* **最好配方** = run #13：`support_dual` + `--scene full035` + `--action-window reach` + `--expl-hold 10` + 起点 B0 + 目标 B1。
  已验证的能力：**一次换手 + 之后双手稳定悬挂约 1.4 s、整集不掉**（位置误差 3 cm，目标距离 1.54→0.095）。
  未达成：`advance_max`=1（推进一根后停住）；双持时间占比 ~0.46；多杆。
* **四个未决问题**：
  1. **单手→双手过渡能否自己走完？**（#13 的趋势仍在上升、无平台 ⇒ 值得更长时间，或 M3.4 起手双持 reset）
  2. **多杆失败是"预算摊薄"还是"跨杆泛化失败"？**（A′：同 #15 配方跑 6 h；B′：`--start-bar-max 1` 只留 2 个任务做 1.5 h 快检）
  3. **`c_sup = max(c_L,c_R)` 的 goal 值不可达**（稳定悬挂实测 ~0.55，目标 1.0）⇒ 每步距离里多一个 ~0.2 的常数，`success` 对"稳定悬挂"永不亮。这是**读数**问题（不影响单/双手差距，差距全在 `h_dual`），修法是把它从 goal 去掉。
  4. **`advance`（自推进目标）从未重跑**：它按设计让 instruction goal 永远只差一根（距离恒 0.6–1.2，始终可达），其唯一已知阻塞（缺方向项）已在 `bf34058` 修好但未验证。
* **环境/工程侧**（与实验设计无关，已在别处记录）：`full035` 资产、`support_dual`/`advance` 变体、双持/滞空指标（§9.40.2）、四个静态检查（`check_args` / `check_metrics` / `check_scene` / `tools/make_patch.sh`）、jaxgcrl vendored 与 G1 网格入库（见 `README.md`、`docs/复现环境.md`）。

## 9.47 run #16（A′：随机起点 + 目标恰好一根，6 h）结果 —— **多任务下连"换手"都没学出来**

`runs/brach_dual_next6h`（60 evals，50.13 M 步，git `1bec3c0`）：与 run #13 相同，只多了 `--start-bar-max 3 --train-goal-bar -1 --goal-ahead 1 --goal-ahead-max 1`（起点 ∈ B0..B3，目标恒为起点后一根）。

| steps | len | fell | `advance_max` | `dual_runmax` | `dual_steps` | `single_steps` | air | `max_bar` | C(0.1) | closest | dist/len | acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.97 M | 501 | 0% | **0** | **488.6** | 498.5 | 2.4 | 0.1 | 0.0 | 0.00 | 0.008 | 0.70 | 0.46 |
| 5.97 M | 501 | 0% | **0** | 37.6 | 184.8 | 304.5 | 9.4 | 0.0 | 0.00 | 0.008 | 1.10 | 0.41 |
| 20.97 M | 158 | 100% | **0** | 3.9 | 4.4 | 131.8 | 14.2 | 7.4 | 1.00 | 0.008 | 1.56 | 0.32 |
| 50.13 M | 138 | 100% | **0** | 1.5 | 1.8 | 83.2 | 21.1 | 13.3 | 0.81 | 0.008 | 1.64 | 0.28 |

**读法（两个容易看错的地方）**：

1. **第一个 eval 的 `dual_runmax = 488.6` 不是技能**：`dual_*` 的判据是"两只手在同一根杆 5 cm 内"——**起点杆也算**。eval 环境被钉在起点 B0（`eval_kwargs["start_bar_max"]=0`），所以 488 步 = **整集吊在 B0 上没动**（同窗 `max_bar=0`、C(0.1)=0 也印证"没往 B1 去"）。§9.41 已经提过这个口径问题，这里是它的极端形式。
2. **`advance_max` 全程 0**：50 M 步里**一次"持续抓稳下一根杆"都没发生** ⇒ 换手技能没有形成。后段 `len` ~140–220、`fell` 100%、`air` 7–25 步、`single` 83–200 ⇒ 收敛到 **run #10 的"掠过后掉"** 模式；`closest = 0.008`（8 mm）说明它**够得到** B1，但 `dual_runmax ≤ 3.9` ⇒ 在 B1 上留不住。critic 健康（`acc` 0.28–0.46、`critic_loss` 3.3）⇒ 不是表征崩。

**结论**：§9.45 里"#15 失败只是预算摊薄（4 个任务 ÷ 12.2 M）"的假设**不成立**——这次给了 50 M 步（每任务 ~12.5 M，与 run #13 相当），**仍然没学出换手**。对比：run #13（单一任务 B0→B1）在 ~6 M 步就出现了换手+双持。⇒ **"起点随机 + 目标随杆"这个多任务设定本身比"单一任务"难得多**，不是单纯加时间能解决的。
⇒ 短期最稳的路径是 **D：run #13 原配方跑 6 h**（先把单杆的"换手 + 双手稳定悬挂"做扎实），多杆再从"只有两个任务（B0→B1、B1→B2，`--start-bar-max 1`）"逐步加。

## 9.48 目标/状态"变粗"的新方案（用户提议）：**只看躯干坐标 + 停在目标杆附近的时长**

用户提议：state/goal 不该那么细（手的位置等），而是取**机器人身体（躯干）的坐标**，取"该坐标的 x、z 停留在下一根/目标杆附近的时间 $S$"，再用 $1-e^{-S/\tau}$ 映射成状态。

**我的评估：这个简化是对的，和我们全部实测证据都吻合**，但有一条必须守住的边界。

### 9.48.1 具体形式

$$g=\big[\;\underbrace{x_{\rm torso}}_{\text{绝对}},\;\underbrace{z_{\rm torso}}_{\text{绝对}},\;\underbrace{1-e^{-S_{\rm park}/10}}_{h_{\rm park}}\;\big]\quad(3\ \text{维})$$

* $S_{\rm park}$ = 连续多少步"躯干位于**某根杆**悬挂点附近的盒子内"（$|x-x_{B_k}|<0.10$、$|z-z_{\rm hang}|<0.15$，**杆号不限**）；
* 目标值 = 目标杆的 $(x,z)$ + $h_{\rm park}=1$（稳定停住）；
* 仍然**每个 bar 一套 goal**（绝对值），按 episode 采目标杆 —— 与 `support` 家族完全一样的机制，所以**可学性有先例**（run #6/#7/#10 都在没有手部维度的 goal 上学会了够杆）。

### 9.48.2 为什么它比现在的 11 维更好（逐条对应我们的证据）

| 好处 | 依据 |
|---|---|
| **有"方向"**：$x,z$ 是**绝对**坐标 ⇒ 每个状态到目标杆都有明确距离 | run #12 的失败正是因为**平移不变**把方向消掉了 |
| **有"持续"**：$h_{\rm park}$ 给的是"停住了"的连续量（自举轴） | run #11：`hold` 是"从擦到变成抓住"的决定性一维 |
| **不可被飞掠伪造**：躯干停在杆下意味着**有支撑**；自由落体穿过盒子只需 ~5 步 ⇒ $S\ge10$（0.2 s）就足够过滤 | run #10 的退化正是"瞬时接触"被当成目标 |
| **没有不可达的维度**：不再有 `c_sup`（稳定悬挂实测只有 ~0.55，目标却是 1.0 ⇒ 每步距离多一个 ~0.2 常数）| §9.41/§9.46.4 |
| **不用手部标定**：绕开"抓手点漂到 3–4.5 cm"、$\tau=0.04$ vs 0.05 之争、以及每只手的距离项 | §9.34/§9.38 |
| **不规定策略**：只描述"身体停在目标杆下方"，一只手/两只手/怎么换手都随策略 | 你的原则（goal 描述终点 ≠ 规定轨迹）|
| **维度低**：3 维 vs 11 维 ⇒ goal 空间小、重标记目标更集中 | §9.45 的"摊薄"现象与 goal 维数直接相关 |

### 9.48.3 必须守住的一条边界 + 会损失什么

* ⚠️ **不要把 $x,z$ 写成"相对下一根杆"**（$x-x_{B_{k+1}}$）。那会重新落回平移不变 ⇒ 方向消失 ⇒ 复现 run #12 的失败。**"目标杆"必须由 goal 的绝对值表达**，而不是由"当前进度"推断。
* 会损失：**"两只手"的要求**（`h_dual`）。停在杆下的躯干位置，单手吊住也能满足。基于实测（#13 的双持占比只有 0.46、而 #6/#7/#10 在没有手部维度的情况下照样学会了够杆）我认为**可以接受**——如果后面发现第二只手确实是瓶颈，再加回**唯一一维** `h_dual` 即可。

### 9.48.4 建议的验证顺序（一次只改一个变量）

1. **单任务**：`--goal-variant park` + 起点 B0 + 目标 B1（与 run #13 同一设定）⇒ 看是否**更快/更干净**地学出"换手 + 稳定停放"（run #13 用了 ~6 M 步出现换手，可作对照）；
2. 通过后**只加起点随机**（`--start-bar-max 3`，目标仍是起点后一根）⇒ 检验跨杆泛化（这正是 run #16 失败的那一步，用来分辨"是 goal 变细太多"还是"多任务本身难"）；
3. 再加**更远的目标杆**（`--goal-ahead 1`，不限距离）⇒ 检验多杆；
4. 需要时再加回 `h_dual`（1 维）。

**实现要点**（`src/envs/brachiation.py`）：新增 goal 变体 `park`（3 维，走 `support` 家族的"每 bar 一套 goal"路径）＋ `info["park_streak"]`（episode 级清零、已进 `episode_info_zero`）＋ 判定盒子参数（`PARK_RX=0.10, PARK_RZ=0.15`）＋ `metrics` 里加 `cov_park_runmax_improve` / `cov_park_on_steps`（评估器白名单与 `check_metrics.py` 同步），并把它加进 `GOAL_VARIANTS` 与 `train.py --goal-variant`。

### 9.48.5 `park` 变体已实现（run #17/#18 待跑）

`--goal-variant park`（3 维，`state=142 / goal=3 / obs=145`，`goal_indices=(0, 2, 127)`）：

```
g = [x_torso, z_torso, h_park],   h_park = 1 - exp(-S_park / 10)
```

`S_park` = 连续"躯干位于**某根杆的悬挂盒**内"的步数（episode 级清零）。每根杆一套绝对
goal（与 `support` 家族同一路径，`goal_set[k] = (hang_x[k], z_hang, 1)`）。

**CPU 实测发现的关键修正 —— 盒子必须锚在躯干悬挂位，不是杆的 x。**
G1 双手勾杆时身体并不在杆的正下方：d035 下 keyframe 躯干 `x = -0.0153`，而
`bar_x[0] = +0.056`，偏移 `dx_goal = -0.071 m`（环境里本就有这个量，`advance` 特征一直
在用）。最初把盒子锚在 `bar_x` 上时，自然悬挂的 `dx` 恰好是 **0.060–0.068**，卡在 ±0.06
边界上，streak 要到 t=21 才开始（前 20 步反复清零）——这正是 M4 文档 §8 提醒的"判据太紧
会让真实稳定悬挂 flicker"。改成 `self.hang_x = bar_x + dx_goal` 后 streak 从 **t=1**
起单调增长，120 步无清零（`dx_hang <= 0.038`、`dz <= 0.042`，对 0.06/0.10 仍有
1.6–2.5× 余量；x 覆盖 `2*0.06/0.35 = 34%` 的杆间走廊，不会把整条摆动通道吞掉）。

| 状态（d035）| `S_park` | `h_park` | `dist` | `success`（<0.18）|
|---|---|---|---|---|
| t=1 吊在 B0，goal=B0 | 1 | 0.10 | 0.905 | 0 |
| t=15 | 15 | 0.78 | 0.225 | 0 |
| t=30 | 30 | 0.95 | 0.064 | **1** |
| t=120（稳态）| 120 | 1.00 | **0.045** | **1** |
| 吊在 B0、goal=B1（60 步）| 60 | 1.00 | **0.315** | **0** |

⇒ `--goal-reach-thresh 0.18` 正好落在 0.045 与 0.315 之间：对"停在对的杆下"留 4× 余量，
对"停在隔壁杆下"留 1.75× 距离。**默认 0.35 绝对不能用**——它恰好等于一格杆距，会把
"停在隔壁杆"也判成成功。

新指标 `cov_park_runmax_improve`（求和 = 最长连续停放段）、`cov_park_on_steps`、
`cov_park_hold_sum`（Σ `h_park`）；评估器白名单（`patches/jaxgcrl_crl_losses.patch`）与
`src/check_metrics.py` 已同步。`train.py` 新增 `--goal-reach-thresh`（默认 0.35 是为 10 维
姿态目标定的，粗目标不要用它）。

一个已知的、无害的读法问题：`h_park` 只在躯干**离开盒子**后才回落，而 episode 在
`z < z_hang - 0.8` 才结束；自由落体走完 0.10 m 盒子下沿约需 7–8 步，所以"松手后头几步"
仍记在 streak 里。因为 `h_park` 是指数累积的持续性量、且成功率由 `dist` 的 `(1-h)` 项
与 `(x,z)` 项共同决定，短暂下落不会伪装成"停稳"（吊着不动 120 步得 `h=1.00`，而任何
真实下落都会在 ~8 步内把 `h` 拉回 0.5 以下并继续掉）。

### 9.49 `park` 放弃 → `dual_nomax`（= run #13 去掉 max()）跑 1.5 h

> ⚠️ **本节关于 `max(c_L,c_R)` 的判断已被 §9.50 实测证伪**：run #18（`dual_nomax`）跑完后同预算明显更差
> （`advance_max`/`succ` 恒 0、`fell` 100%），而 CPU 实测显示 `c_sup` **并不饱和**（稳态悬挂只有 0.52），
> 且在「已吊在目标杆下」的状态里占剩余距离的 **93%** —— 它是承重项。下面的设计动机保留作过程记录，
> **结论以 §9.50 为准**。

**用户判断（推翻 §9.48/§9.48.5）**：`park` 的「悬挂盒」思路不合理——盒子的锚点和尺寸都必须人为拍定（§9.48.5 里我为了让它不 flicker，把锚点从 `bar_x` 改到 `hang_x`、把尺寸从 0.10/0.15 收到 0.06/0.10，两次都是人工调参），而**手部接触与手部末端位置在现实中本来就是可测量/可估计的**（关节编码器 + FK）。所以粗目标带来的「盒子难定」是纯粹的额外负担，取消；`park` 代码保留仅供复现。

同时确认：**手部末端位置应当保留**。`full`/`support` 目标里的 `p_L/p_R` 是 `_hand_points()` 给出的**每只手的抓握座世界坐标**（`seat_local` 在 keyframe 标定、由腕部 `xpos/xmat` 变换得到）；keyframe 下两手都勾在 B0，所以两个座标都等于杆点 `(0.056, 0, 1.486)`——这不是冗余，而是「每只手的抓握座落在第 k 根杆上」这一连续、有方向、可测量的目标项（§9.49 实测确认）。

**新变体 `dual_nomax`** = run #13 的 `support_dual` **只去掉一维**（瞬时 `max(c_L,c_R)`，superset 下标 10）：

```
g = [x_torso, z_torso, p_L(3), p_R(3), h_any, h_dual]        10 维, state 144, obs 154
goal_indices = (0, 2, 99..104, 128, 129)   # 128/129 = hold / hold_dual，跳过 127 = c_sup
```

`max(c_L,c_R)` 是三个候选里最差的一维：饱和（挂着就是 1，没有方向）、稳挂读 ~0.55 对目标 1.0 ⇒ 给每个距离贡献 ~0.2 常数（§9.41 的 unreachable c_sup 读数问题）。`h_any`/`h_dual` 是**持续**接触特征：抓住时增长、松手即清零，既有方向又不需要任何手调盒子。`max(c_L,c_R)` 仍留在 **state** 里（可观测量，critic 能看到），只是不进 goal。默认 `--goal-reach-thresh 0.35` 不需要改。

CPU 校验（d035，吊在 B0 不动）：

| goal | t=15 | t=25 | t=30 | success(<0.35) |
|---|---|---|---|---|
| **B1**（目标杆）| 0.695 | 0.629 | 0.621 | 0 |
| **B0**（正吊着的杆）| 0.320 | — | **0.095** | **1** |

⇒ 0.095 与 0.63 之间留出 6.6× / 1.8× 的余量，阈值 0.35 恰好居中，因此这一轮与 run #13 是**同预算、同阈值、同场景的干净 A/B**（唯一差别 = 目标少一维）。

守卫：`check_args` / `check_metrics` 通过；11 个变体全部构造成功且 `goal 维数 == goal_set 宽度`（`dual_nomax` goal 10 / obs 154 / idx 跳过 127）。`render_policy.py` 的 `uses_dual` 已扩展到 `("support_dual", "dual_nomax")`。

### 9.50 结果：#17（`support_dual` 6 h）成功；#18（`dual_nomax`）证伪

两个 run 都跑完并同步进仓库：`runs/brach_dual_b1_6h/`（#17，60 evals / 5.013e7 步）、
`runs/brach_dual_nomax_b1/`（#18，20 evals / 1.22e7 步）。

**读表须知**：`eval/episode_dist` 是 **episode 内逐步求和**，不能直接横向比（#18 的 178 < #13 的 393
只是因为它死得早、集数短）。要看 `dist/len`，其中 `len` = `eval/avg_episode_length`（我上一轮查错了列名）。

#### 9.50.1 同预算对照（1.22e7 步 / 20 evals）

| 指标 | #13 `support_dual` | #18 `dual_nomax` | #17 @20 | #17 @60（5.01e7）|
|---|---|---|---|---|
| `dist/len` | 0.830 | 1.389 | 0.399 | **0.198** |
| `dual_runmax`（最长同杆双持，步）| 69.8 | **9.8** | 314.9 | **424.6（8.5 s）** |
| `dual_steps` | 219.9 | 10.7 | 374.6 | 472.2 |
| `single_on_steps` | 209.4 | 79.2 | 57.6 | 15.9 |
| `bar_L` / `bar_R`（均值 = 杆号）| +0.79 / **−0.08** | −0.77 / −0.37 | +0.81 / +0.73 | **+0.89 / +0.86** |
| `max_bar`（均值）| 0.97 | **0.12** | 0.95 | 0.94（= B1）|
| `advance_max`/len | 0.83 | **0.00** | 0.88 | 0.89 |
| `fell` | 6% | **100%** | 12% | **0%** |
| `succ`/len | 0.32 | **0.00** | 0.69 | **0.83** |
| `hand_on_steps`/len | 0.78 | 0.47 | 0.90 | 0.95 |
| 最近距离（`min_d_b1`）| 0.002 | **0.018** | — | **0.000** |

#### 9.50.2 #18 是证伪，不是「略差」

同预算下 `advance_max` 与 `succ` **恒 0**、`fell`=100%、`max_bar` 均值仅 0.12。它并不是「够不到」：
`C(0.1)`=1、最近距离 **1.8 cm** ⇒ 能掠过 B1，但**从不在 B1 的抓握窗内停留**（`cov_b1_runmax`=0，
`max_bar` 只有约 16 步为 1）。更关键的是形态：eval 1 时它本来是**完美的双手悬挂**
（`dual_runmax` 490.8 = reset 状态），随后训练**把它拆掉** —— eval 5 `fell` 94%、eval 10 短暂回到
len 409 / `dual_steps` 212、eval 15 又落回 `fell` 100% 并保持到结束。而 #13 与 #17 在同一阶段
（eval 12–15）恰恰是**开始起飞**的转折点。⇒ 去掉 `max(c_L,c_R)` 的配方锁不住「抓住」，不稳定。

#### 9.50.3 §9.49 的机理判断被实测推翻

§9.49 说 `max(c_L,c_R)` 是「饱和、无方向、贡献恒定 ~0.2 的拖累项」。CPU 实测（稳态悬挂 25 步，
`h_any`=`h_dual`=0.918，目标杆 = B1）逐维分解：

| 状态 | `c_sup` 实测 | 该维贡献 | 占 dist² 比例 |
|---|---|---|---|
| 吊 B0、目标 B1（`support_dual`）| **0.524**（目标 1.0）| 0.226 / 总 0.622 | **36.4%** |
| 吊 B0、目标 B0（已吊在目标杆下）| 0.524 | 0.226 / 总 0.243 | **93.0%** |
| 对照：`dual_nomax` 的全部 10 维 | — | 0.396 | — |

即 `c_sup` **根本不饱和**：真实悬挂下只有 **0.52**（因为它对 2–4 cm 的抓握座偏移极敏感，τ=0.04 m），
而且当机器人已经吊在目标杆下时，剩余距离的 **93%** 来自这一维。它是「抓握质量」的主项，不是可以删掉的
饱和项。删掉之后目标被纯几何项主导（x / p_L / p_R 分别占 28% / 34% / 33%），训练不再锁得住「抓住」。
同族现象：run #12（`advance`，无接触项）一步没动、`acc` 0.08。

#### 9.50.4 #17 = M4 目前最好的策略

- `dual_runmax` 69.8 → **424.6 步 = 8.5 s 连续双持**；`fell` 6% → **0%**；`succ/len` 0.32 → **0.83**；
  `dist/len` 0.83 → **0.198**；`hand_on_steps` 78% → **95%**。
- `bar_R` 从 **−0.08（右手几乎从不进抓握窗）→ +0.86**：**单手侧吸引子被解决**，两只手都稳定停在
  **B1**（`bar_L` +0.89、`bar_R` +0.86、`max_bar` = B1、`advance` 平台 = 1）。
- `air_runmax` = 10.6 步（**0.21 s**）⇒ 转移只有极短腾空，是**受控换手**，不是 run #10 那种
  「双手离杆鱼跃」。
- 单调性：`single_on_steps` 57.6→15.9 与 `dual_steps` 374.6→472.2 互为镜像 ⇒ 是「右手补上来」，
  不是「左手掉下去」。
- eval 20 已起飞（`dualMx` 314.9 / `fell` 12%），eval 30 后进入平台（len = 501 满、`fell` 0、
  `dualMx` 409–425）⇒ **固定目标下「6 h 预算被稀释」不成立**；run #16 的失败属于「随机起点 + 多目标」
  那个设置，不能推广成「时间不够」。

#### 9.50.5 结论与下一步

1. **保留 `c_sup`**（承重项）；`dual_nomax` 归档为证伪，不再投 6 h。
2. M4 推进必须**单变量**（#16 的教训 = 别同时动起点和目标）：固定起点 B0、`support_dual` 配方不动，
   只把目标杆往前推一格（`--train-goal-bar 2 --eval-goal-bar 2`，或 goal ∈ {1,2} 采样），
   检验「连续过多杆」能否从 1 格扩到 2 格。
3. 先用 `render_policy.py` 回放 #17 的 `actor_latest.pkl` 确认策略形态（受控换手的具体方式），
   再决定多杆目标怎么给。

### 9.51 为什么删 `max(c_L,c_R)` 会崩（更深的分析）+ 改成 `c_L,next, c_R,next`

#### 9.51.1 直接证据：把 #18 的 ckpt 回放出来

`runs/ckpt_dual_nomax/actor_latest.pkl`，`--goal-bar 1`，2 集 × 501 步（与 #17 同一套回放）：

| | #18 `dual_nomax` | #17 `support_dual` 6h |
|---|---|---|
| 存活 | `done@128` ⇒ **2.5 s 就掉下去** | 全程不掉 |
| 两手杆号 | `(-1,-1)` **412/501 步（82%）**；(1,1) **0 步** | `(1,1)` **443/461 步（88–92%）** |
| `c_L` / `c_R` 均值 | **0.03 / 0.11** | **0.88 / 0.87** |

⇒ #18 学到的不是「不会转移」，而是「**松手往下掉**」：82% 的时间两只手都不在任何杆上。
所以这不是「差一点」，是目标被改掉以后**学到的行为方向变了**。

#### 9.51.2 被删掉的那一维到底有多重（实测增益）

`c = exp(-(d/TAU_CONTACT)^2)`，`TAU_CONTACT = 0.04 m`。目标里的 `c` 值是 **1.0**——它就是
keyframe 的对齐值（`seat_local` 正是按「t=0 时抓握座与杆心重合」标定的）；而**稳态悬挂**时抓握座
会松弛到离杆心 **3.2 cm**：

| goal 维 | 目标值 | 稳态悬挂实测 | 缺口 | 同样 3.2 cm 误差折算成该维缺口 |
|---|---|---|---|---|
| `x,z,p(6)`（米制，8 维）| keyframe 值 | 差 3.4 cm（z）/ 3.2 cm（座）| 0.03–0.034 **每维** | 0.001–0.0012 |
| `max(c_L,c_R)` | 1.0 | **0.52** | **0.48** | **0.226（增益 ≈ 15 /m）** |
| `h_any` / `h_dual` | 1.0 | 0.918 | 0.082 | —（10 步时间常数）|

`max(c)` 是整个目标里**增益最高**的一维：同一个 3.2 cm 误差，在它这里是 0.226，在米制位置维里只有
0.0012（差约 200 倍）。它的物理含义是「**把抓握座从松弛的 3.2 cm 拉回杆心**」——而 #17 的策略确实做到了
**0.96–0.98**（9.51.1 实测），说明它是**可优化、并且被优化掉了**的一维。

删掉它 = 去掉整个目标里最密、最局部、增益最高的塑形信号，只剩 0.35 m 尺度的几何项和 10 步时间常数的
持续项；训练于是崩向「松手坠落」。§9.49 说的「饱和 / 无方向 / 恒定 ~0.2 拖累」三条全部不成立：
它不饱和（0.52→0.98，可变范围 0.46）、方向明确（趋近杆心）、而且在「已吊在目标杆下」时占剩余距离 **93%**。

#### 9.51.3 用户提议：`max(c_L,c_R)` → `c_L,next, c_R,next`

已实现为变体 **`dual_cnext`**（12 维，`state 146 / obs 158`）：

```
g = [x, z, p_L(3), p_R(3), c_L,gk, c_R,gk, h_any, h_dual]
c_h,gk = exp( -(d[h, gk] / TAU_CONTACT)^2 )        # 每只手 × 指令目标杆 gk
goal_indices = (0, 2, 99..104, 130, 131, 128, 129)
```

`gk` 是**每条 episode 固定的指令目标杆**（新增 `info["k_goal"]`，由 `reset()` 写入；`_cnext()` 是
`_state_features` 与 `_achieved_goal` 的单一实现，保证状态与目标永不打架）。用「指令目标杆」而不是
「`k_ref+1`」是因为目标向量必须在 episode 内**平稳**：`k_ref` 在转移成功后会跳到下一根，那样 `c_next`
的含义会在 episode 中途改变。

为什么这比 `max(c_L,c_R)` 好：

1. **保住了那个承重的高增益接触项**（9.51.2 的增益 ~15/m 完整保留），所以不会重演 #18 的崩塌；
2. 把「任意一只手 / 任意一根杆」换成「**每只手对目标杆**」：挂在起始杆上**不再部分满足**——CPU 实测
   吊在 B0、goal=B1 时 `c_L,next = c_R,next = 0.000`（两维各占总距离 41.7%），而同一时刻
   `max(c_L,c_R)` 是 0.52、缺口只有 0.476；
3. 单手也不再够（两维必须都到 1）；
4. 它自带**分步结构**：先一只手抓住目标杆 `(1,0)`，再第二只手 `(1,1)`——正是 CRL 的未来状态重标记
   最需要的中间态；而 `max()` 版本里「抓到目标杆」与「抓在起始杆上」不可区分。

CPU 实测（d035，稳态悬挂 25 步）：

| goal | `c_L,next` | `c_R,next` | 这两维占比 | `abs(diff)` |
|---|---|---|---|---|
| **B1**（吊在 B0）| **0.000** | **0.000** | **83.4%** | **1.548** |
| **B0**（已吊在目标杆下）| 0.474 | 0.524 | 96.7% | **0.721** |

⚠️ **阈值必须改**：默认 `--goal-reach-thresh 0.35` 对 `dual_cnext` 太紧——连「已经吊在目标杆上」
都是 0.721 ⇒ `success` 会恒 0。用 **`--goal-reach-thresh 0.8`**（0.72 < 0.8 ≪ 单手/起始杆的 ~1.5）。

保留 `h_any`/`h_dual` 是刻意的：这一轮是**只替换那一维**的单变量实验；「把两个持续项也去掉」留作下一步
再单独测。

#### 9.51.4 run #19 侦察命令（1.5 h，与 #13 同预算，可逐 eval 叠曲线）

```bash
.venv-warp/bin/python src/check_args.py && .venv-warp/bin/python src/check_metrics.py

.venv-warp/bin/python -u src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant dual_cnext --goal-reach-thresh 0.8 \
  --train-goal-bar 1 --eval-goal-bar 1 --expl-hold 10 \
  --num-evals 20 --steps 12200000 --save-every 5 \
  --checkpoint-dir runs/ckpt_dual_cnext --impl warp --scene full035 --gpu 0 \
  --wandb --exp-name brach_dual_cnext_b1 2>&1 \
  | grep --line-buffered -v dot_search_space | tee runs/brach_dual_cnext_b1.log
```

判据（与 #13/#18 同预算对照）：`advance_max`/`max_bar` 是否越过 B1、`dual_runmax` 是否显著超过 #13 的
69.8 步、`fell` 是否远低于 100%，以及 `bar_L`/`bar_R` 是否都转正（两手都到 B1）。

#### 9.51.5 澄清：「接触」在这个仓库里是三个不同的量

用户提问「c 不是接触吗，难道不是二值？」——不是同一个东西，前面我的措辞容易混：

| 量 | 定义 | 性质 | 用途 |
|---|---|---|---|
| `bar_L/bar_R/max_bar`、`hold_streak`/`dual_streak` 的判据 | `d < GRASP_THRESH = 0.05 m` | **空间二值**（每步 0/1）| 指标读数 + 持续计数。M4 文档 §8 定的：用 5 cm，不用 3.3 cm（真实悬挂会漂到 3–4.5 cm）|
| `h_any` / `h_dual` | `1 - e^{-S/HOLD_TAU}`，S = **连续满足上式**的步数 | **时间连续**（空间仍二值），`HOLD_TAU`=10 步 | goal 维：「抓住了并且一直抓着」|
| `c`（特征 `grasp_ind`，以及 goal 里的接触项）| `c = exp(-(d/TAU_CONTACT)^2)`，`TAU_CONTACT` = 0.04 m | **空间连续**、可微、单调 | goal 维：「抓握质量」（瞬时接触）|

`d -> c`：0 cm 1.00、1 cm 0.94、2 cm 0.78、**3.2 cm 0.53**、4 cm 0.37、
**5 cm（抓握窗边界）0.21**、10 cm 0.002。⇒「在 5 cm 窗内」这个二值判据在 `c` 里对应
**0.21 → 1.0 的一段连续质量分**。稳态悬挂实测 `c = 0.527`（反解 d = 3.23 cm）；run #17 的策略把
`c` 做到 0.96–0.98（反解 d ≈ 0.7 cm）——它是真的把自己往杆上拉。

**为什么 goal 里必须用连续版**：布尔维度对状态的梯度几乎处处为 0（跨过阈值前目标距离恒定），actor
拿不到方向，这正是 run #8 `cross3` 的退化原因（§9.31）。**增益**：`dc/dd = -2 d c / tau^2`，在
d = 3.2 cm 处 ≈ **-21 /m**，而米制位置维是 1 /m ⇒ 同一个 3.2 cm 误差在 `c` 里折算 0.48、在 `p` 里
只有 0.032（§9.51.2 的表由此而来）。

**正确读法**：`max(c_L,c_R)` = 「最好那只手的抓握质量」；`c_L,next / c_R,next` = 「左/右手的抓握座
**对目标杆**的抓握质量」；目标值 1.0 = 抓握座与目标杆心完全重合（keyframe 的理想姿态，`seat_local`
就是按这个标定的）。旁注：`TAU_GRASP = 0.05`（`_grasp` 里 softmax「是哪根杆」的温度）与
`GRASP_THRESH = 0.05`（窗口）数值相同但作用不同，和 `TAU_CONTACT = 0.04`（软接触宽度）一起，
三个常数都是 cm 量级。

### 9.52 「全摊开」方案 `dual_hnext` + 一串基础事实的核对（用户提问）

#### 9.52.1 goal 里的实际取值（代码核对，别再靠记忆）

| 量 | 指令 goal（`goal_set[k]`）里的值 | 训练时（relabelled）|
|---|---|---|
| `x_torso` | `bar_x[k] - 0.0709`（**不是**杆的 x，是**躯干**悬挂时的 x）| 未来状态的 `qpos[0]` |
| `z_torso` | `0.7743`（keyframe 躯干高度，**每根杆都一样**）| 未来状态的 `qpos[2]` |
| `p_L/p_R` | `bar_x[k]`（抓握座标定在杆心，所以正好等于杆的 x）| 未来状态的座标 |
| `c_*` | **1.0**（keyframe 对齐；稳态悬挂只有 0.47–0.52）| 未来状态的 `c` |
| `h_any/h_dual` | **1.0**（「已经持续抓住」）| 未来状态的 `h` |

实测：`goal x = [-0.0153, 0.3347, 0.6847, 1.0347, 1.3847]`，`bar_x = [0.0556, 0.4056, …]`，
差值恒为 **-0.0709** ✓；`goal z` 恒为 0.7743 ✓。稳态悬挂实测 z = 0.7405（低 3.4 cm）、座离杆心 3.2 cm
⇒ 这就是 §9.50/§9.51 里那些 0.03/0.48 缺口的来源。

#### 9.52.2 三种「持续接触」的区别（用户提问）

| 量 | 判据（连续 S 步）| 强度关系 | 会不会在**起始杆**上就满足 |
|---|---|---|---|
| `h_any` | ≥1 只手在**某根杆**的窗内 | 最弱 | **会**（吊 B0 时 0.918）|
| `h_dual` | 两只手在**同一根杆**的窗内 | 中 | **会**（同杆 = B0）|
| `h_L` / `h_R`（每手各一个）| 该手在**某根杆**窗内 | 中（`h_dual ⇒ (h_L,h_R) ⇒ h_any`）| 会 |
| **`h_L,gk` / `h_R,gk`（本轮新增）** | 该手在**指令目标杆**的窗内 | 最强且**指向目标** | **不会**（吊 B0、goal B1 时 = 0）|

前两者的“持续”是**杆无关**的，所以在起始杆上就已经接近满分（各 0.918），也会奖励「原地等着 h 长大」；
`h_*,gk` 把这两条漏洞一起堵掉。

#### 9.52.3 新变体 `dual_hnext`（用户的「全摊开」方案）

```
g = [x, z, p_L(3), p_R(3), c_L,gk, c_R,gk, h_L,gk, h_R,gk]       12 维, state 148 / obs 160
goal_indices = (0, 2, 99..104, 130, 131, 132, 133)
S_h  = 第 h 只手连续在 **gk 号杆** 5 cm 窗内的步数；h_h,gk = 1 - e^{-S_h/10}
```

`h_any`/`h_dual` **保留在 state 里**（可观测量 + 指标），只是**不进 goal**。CPU 实测（t=25，吊在 B0）：

| goal | `c_L,gk` | `c_R,gk` | `h_L,gk` | `h_R,gk` | 四维占比 | `abs(diff)` |
|---|---|---|---|---|---|---|
| **B1**（吊在 B0）| **0.000** | **0.000** | **0.000** | **0.000** | **91.2%** | **2.093** |
| **B0**（吊在目标杆下）| 0.474 | 0.524 | 0.918 | 0.918 | 99.3% | **0.721** |

对比 `dual_cnext` 在同一状态（吊 B0 / goal B1）：`c` 两维为 0，但 `h_any`/`h_dual` 已是 0.918，各只占
**1.5%** ⇒ 那两个「杆无关」维在错误杆上几乎白送。`dual_hnext` 把它们换成 0.000，目标在错杆上
**没有任何一维是接近满足的**。

**评估**：
- 正面：① 终态表述变得完全明确（**每只手各自持续抓住目标杆**）；② 消除「等待 h 长大」的被动捷径与
  「错杆白送」；③ `(c_L=1,c_R=0,h_L↑,h_R=0)` 这种**单手已抓住目标杆且持续**的中间态变得可表达，
  正是 CRL 未来状态重标记最需要的量；④ 右手停在 B0 时 `h_R,gk` 恒 0 ⇒ **单手侧吸引子被直接惩罚**。
- 代价/风险：目标**更难**（两维要求对目标杆持续 30+ 步），早期可能更容易掉进 evals 3–12 那个崩塌期；
  引导必须靠 `c_*,gk` + `p(6)`（都是指向目标杆的）——这也是为什么不该同时把 `p(6)` 拿掉。
- 阈值：同样要用 `--goal-reach-thresh 0.8`（吊在目标杆下实测 0.721；错杆 2.093）。
- `x/z` 绝对值 vs 相对值：**在固定 B0→B1 下完全等价**（只差一个仿射平移），同意「没什么区别」；
  绝对值只在放开随机起点时才重要（杆身份），而 run #12 的平移不变失败是前车之鉴 ⇒ 保留绝对值。

#### 9.52.4 还没被证伪的一条：`max()` 会不会只是「学得慢」

严格说，目前**不能**排除「给 #18 足够探索它也能学会过杆」：
- 只有 1 个 seed，且两个成功 run（#13、#17）**同样**在 evals 3–12 崩过（`fell` 100%、len 40–130），
  到 eval 12–15 才起飞；#18 也在 eval 10 短暂回到 len 409 / `dual_steps` 212，随后又落回去。
- 能确定的只是「同预算（1.22e7 步）下 #18 远差、且末期趋势向下、`cov_b1_runmax` 恒 0」。
- 另外要澄清一点：#18 的「伸手去够下一个杆」**不是** `h_any` 驱动的（`h_any` 只要求某根杆，
  吊在 B0 就已经满足），而是 `x/z/p(6)` 这些**指向 B1 的几何维**驱动的——它确实够到了 B1 附近
  （最近 1.8 cm），但从不停留。缺的不是「向目标迁移的量」，而是**让「抓住」这件事本身值得做的那一维**。
- 想判定「慢还是不可能」，最干净的是把 `c_*,gk` 换成**线性**的每手到目标杆距离（`dual_dnext`，信息
  完全相同、只去掉非线性/高增益）：能学会 ⇒ 关键是「每手 × 目标杆」这个引用方式；也崩 ⇒ 增益才是关键。

#### 9.52.5 两臂对照命令（都与 #13 同预算 20 evals / 1.22e7 步）

```bash
.venv-warp/bin/python src/check_args.py && .venv-warp/bin/python src/check_metrics.py

# GPU 0：只换 c（每手 × 目标杆），h 仍是 h_any/h_dual
.venv-warp/bin/python -u src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant dual_cnext --goal-reach-thresh 0.8 \
  --train-goal-bar 1 --eval-goal-bar 1 --expl-hold 10 \
  --num-evals 20 --steps 12200000 --save-every 5 \
  --checkpoint-dir runs/ckpt_dual_cnext --impl warp --scene full035 --gpu 0 \
  --wandb --exp-name brach_dual_cnext_b1 2>&1 \
  | grep --line-buffered -v dot_search_space | tee runs/brach_dual_cnext_b1.log

# GPU 1：c 与 h 都摊开（h_any/h_dual → h_L,gk/h_R,gk）
.venv-warp/bin/python -u src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant dual_hnext --goal-reach-thresh 0.8 \
  --train-goal-bar 1 --eval-goal-bar 1 --expl-hold 10 \
  --num-evals 20 --steps 12200000 --save-every 5 \
  --checkpoint-dir runs/ckpt_dual_hnext --impl warp --scene full035 --gpu 1 \
  --wandb --exp-name brach_dual_hnext_b1 2>&1 \
  | grep --line-buffered -v dot_search_space | tee runs/brach_dual_hnext_b1.log
```

判据：`advance_max`/`max_bar` 是否越过 B1、`dual_runmax` 是否超过 #13 的 69.8 步、`fell` 是否远低于
100%、`bar_L`/`bar_R` 是否**都**转正。

#### 9.52.6 `x_torso` 绝对 vs 相对（用户追问，补完整）

「相对」必须先说清参照物，一共四种：

| 参照物 | `goal x` 在每根杆上的取值 | goal 是否携带杆身份 | episode 内是否平稳 |
|---|---|---|---|
| (1) **绝对**（世界系，当前实现）| `[-0.0153, 0.3347, 0.6847, 1.0347, 1.3847]` | **是** | 平稳 |
| (2) 相对**起始杆** `bar_x[k0]` | `[-0.0709, 0.2791, 0.6291, 0.9791, 1.3291]` | **是** | 平稳 |
| (3) 相对**目标杆** `bar_x[gk]` | `[-0.0709, -0.0709, -0.0709, -0.0709, -0.0709]` | ❌ 否 | 平稳 |
| (4) 相对**当前手所在杆** | 参照物在 episode 内会跳 | ❌ 否 | ❌ **不平稳** |

**数学事实（决定性的那条）**：只要参照物在 episode 内是常数 `c`，特征与目标**同时**减去 `c`，
距离**逐位不变**：`|(x-c) - (x*-c)| = |x - x*|`。而 `gk`（以及 `k0`）在一个 episode 内恒定，
所以 (1)(2)(3) 在本 episode 内**完全等价**——指令 goal 与训练用的重标记 goal 都一样（重标记取的是
同一 episode 未来状态的目标切片，同样是常数平移）。

实测（`position` 变体，t=25 吊在 B0，goal=B1，状态 `x=-0.0008 z=0.7405`，目标 `x=0.3347`）：

| 参照物 | `abs(diff_x)` | `abs(diff)` |
|---|---|---|
| (1) 绝对 | 0.335577 | 0.618350 |
| (2) 相对起始杆 | 0.335577 | 0.618350 |
| (3) 相对目标杆 | 0.335577 | 0.618350 |

三者**完全相同**。

**所以唯一会变的事情只有一件**：当**目标杆跨 episode 变化**时（`--train-goal-bar -1` 或
`--goal-ahead`），(1)(2) 的 `goal x` 随杆变化 ⇒ goal **携带「去哪根杆」**；(3) 对所有杆都是同一个
值 ⇒ goal 不再指出目标杆，只剩状态里的绝对位置还能区分 ⇒ 这正是 run #12（`advance`，全部相对、
`acc` 0.08、一步没动）那类退化的来源。计划 §4 早已写明：`p_L,p_R` 用**世界坐标**就是为了让 goal
隐含「抓的是哪根杆」，不必再额外编码 bar index。

**`z` 是特例**：5 根杆等高（`bar_z = 1.4857`），`goal z = 0.7743` 对每根杆都相同 ⇒ 相对化只是常数
平移，**任何情况下都没有区别**（除非将来杆不等高）。用户「z 取绝对/相对没区别」的直觉正确。

**结论**：
- 固定 B0→B1（我们当前所有 run）下，绝对 / 相对起始杆 / 相对目标杆 **完全等价，改不改都一样**；
- (4) 因非平稳直接排除（`k_ref+1` 那类参照物同理）；
- 只有放开随机起点或多杆时才产生实质差别，而那时应当**保留绝对 x（或等价的「相对起始杆」）**：
  goal 里的杆身份正是防止 run #12 式退化的东西。

**旁注（顺带修掉的两个误导性打印）**：`render_policy.py` 的汇总行原本硬编码
`(10-dim)` 与 `(bar spacing 0.4 m)`，对 `support_dual`（11 维）和 `full035`（0.35 m）都会报错值，
已改为按 `len(env.goal_indices)` 与 `env.bar_spacing` 动态取。

### 9.53 多杆扩展：`_next` 的失效点，以及「枚举所有杆」是否合理（用户提议）

#### 9.53.1 先划一条概念分界：视觉枚举属于 **state**，不属于 goal

「像人用视觉识别所有杆」这件事，对应的是**观测**，而我们的观测**已经枚举了所有杆**：
`d_{hk}`（2×5 = 10 维，每只手到每根杆的距离）+ `w_{hk}`（2×5 维 softmax，计划 §3.2 的核心设计，
"挂住时接近 one-hot、飞行途中平滑过渡，同时给出挂在哪根杆"）。而人的**目标**并不是"一张接触图"，
而是"去那一根"＋隐含的"放开旧的"。所以这个类比支持"每根杆的信息进 state"，本身**不推出**"每根杆的
接触进 goal"。

但要注意：这个提议**并不离谱**——它正是计划 §1.4 列出的**最激进选项**：

> 最激进的做法是完全照 Single-Goal CRL：直接提供一帧最终 simulator observation $g^*=o_{\rm final}$

把接触图放进 goal，就是"用一帧最终观测（的接触切片）当目标"。它处在这个谱系的极端一端，
代价与收益都要算清楚（见下）。

#### 9.53.2 `_next` 在多杆下确实会坏（用户判断正确）

单一参照物 `gk` 有两个硬伤：

1. **中间杆没有信号**：B0→B2 时，"一只手在 B1 上抓一把"对 `c_*,gk`/`h_*,gk` 毫无贡献 ⇒ 串杆所需的
   中间态无法被表达、也无法被重标记成 goal；
2. **goal 无法表达「松开启始杆」**——而计划的核心洞见恰恰是
   「$\boxed{\text{为了前进，机器人必须主动放弃一个稳定接触}}$」。

另外一条硬约束：**参照物必须是 episode 内的常数**（`gk`、`k0` 都是），**绝不能用 `k_ref`**——它在转移
成功后会跳到下一根，目标向量的语义会在 episode 中途改变。所以平稳的选择只有两类：
**全枚举**，或**以 `k0` 为锚的窗口**。

#### 9.53.3 数字评估：全枚举买到了什么，代价多少

用实测值（抓握座离杆心 3.2 cm ⇒ `c ≈ 0.474/0.524`；持续 25 步 ⇒ `h ≈ 0.918`；几何项由 §9.52 的
逐维差值平方和给出 0.383）：

| 设计 | 吊在 B0（**必须离开**）| 吊在 B1（**已到达**）|
|---|---|---|
| **A** 只引用目标杆 `gk`（12 维，当前 `dual_hnext`）| **2.094** | 0.721 |
| **B** A + 起始杆接触×2（14 维，goal = 0）| 2.210 | 0.721 |
| **B2** A + 起始杆持续×2（14 维，goal = 0）| 2.463 | 0.721 |
| **C** 全枚举 4n+8 = **28 维**（其余杆 goal 全 0）| **2.563** | 0.721 |

**C 相对 A 的边际只有 +2.185 的平方距离**，拆开是：

* 起始杆的**接触**项 `c_L,k0² + c_R,k0²` = **0.499** ← 这才是"松手"的直接压力；
* 起始杆的**持续**项 `2·h²` = **1.685** ← 与 c 项**冗余**（都在说"你还抓着 B0"）；
* B2/B3/B4 的项全为 **0**（手根本不在那儿）⇒ **28 维里有 8 维是纯常数**。

所以：**"松开启始杆"这个全枚举最有价值的新信号，用 2 维（B）就能拿到**，而全枚举多出的 10 维里
有一半是冗余的持续项、另一半是常数维。在**当前 B0→B1/B2 场景下，C 与"以 k0 为锚的 3 根杆窗口"
（20 维）完全等价**。

#### 9.53.4 全枚举真正独有的能力（不能只用 B 替代）

它能让**重标记的 goal 命名任意中间姿态**，例如「L 手在 B1、R 手仍在 B0」。B 只有两个参照物
（`k0` 与 `gk`），当 `gk = k0+2` 时**无法命名 B1**——这正是"连续过多杆"最需要的中间态。

因此正确的形态不是"无条件全枚举"，而是**以 `k0` 为锚、覆盖到 `gk` 的窗口**：
`4w + 8` 维（`w` = 起点到目标之间需要覆盖的杆数）。对 `gk ≤ k0+2` 用 `w = 3` ⇒ **20 维**，
与全枚举等价但省掉 8 个常数维；只有目标可能到 B4 时才需要 `w = 5`（28 维）。

#### 9.53.5 代价与风险

* goal 维度膨胀（28）⇒ goal 会成为 obs 里最大的一块（state 148 + goal 28）；
* 大量 0/常数维会**稀释指标可读性**（`dist`/`success` 更难解释）——这正是 `c_sup` 那次让我们误判的
  同类问题；
* 对比任务的维度更高、更难（我们 evals 3–12 那个崩塌期本来就脆）；
* 但收益是实的：把"放弃旧接触"变成**显式的高增益目标**（+0.499），而当前所有变体都**没有**这一项。

#### 9.53.6 建议：分三步、单变量

1. **已提交**：`dual_hnext`（每手 × 目标杆的 c/h，12 维）。
2. **下一步（+2 维，最便宜）`dual_origin`** = `dual_hnext` + 起始杆的 `c_L,k0, c_R,k0`（goal = 0）：
   显式"放弃旧接触"。参照物 `k0` 是 episode 常数 ✓。先看它能否让转移更快更稳。
3. **若要串杆 `dual_map`**：以 `k0` 为锚、`w` 根杆的**全枚举**（`4w+8`，`w=3` ⇒ 20 维），
   用它命名中间杆上的姿态；`w` 随目标距离增长。**不要**一上来就用 5 根杆 28 维（8 个常数维）。
4. 保留指令仍是**单一下标** `gk`（目标杆），枚举图是从 `gk` 派生的 ⇒ 不引入指令歧义。

结论：**方向对，但"同时枚举所有杆"里的持续项是冗余的、参照物必须是 `k0`/`gk` 这类 episode 常数；
优先做"锚在 `k0` 的窗口"，而不是无条件 4n 全枚举。**

### 9.54 重新梳理 g（6 维）+ 当前观测逐块清点

#### 9.54.1 当前观测逐块（`dual_hnext`，直接读 `FeatureLayout`，不凭记忆）

| 块 | obs 下标 | 维 | 是否 goal | 备注 |
|---|---|---|---|---|
| root 位置 x,y,z | 0:3 | 3 | **x,z 是 goal** | y 只有初始噪声（平面内运动），无任务信息 |
| root 姿态四元数 | 3:7 | 4 | context | 4 维表示 3 自由度（1 维冗余）|
| root 线速度 | 7:10 | 3 | context | |
| root 角速度 | 10:13 | 3 | context | |
| 关节位置 q[7:] | 13:56 | 43 | context | **含 14 个指关节**，但没有任何 goal/接触特征读它 ⇒ "手指是否合上"只在 q 里隐式存在 |
| 关节速度 dq | 56:99 | 43 | context | |
| 两手抓握座 p_L(3),p_R(3) | 99:105 | 6 | **goal** | 世界坐标；计划 §4 说它隐含"抓的是哪根杆" |
| 每手到每杆距离 d[2,5] | 105:115 | 10 | context | |
| 每手对每杆 softmax w[2,5] | 115:125 | 10 | context | |
| 每手软接触 c_L,c_R | 125:127 | 2 | context | |
| `max(c_L,c_R)` | 127:128 | 1 | context | |
| `h_any` | 128:129 | 1 | context | |
| `h_dual` | 129:130 | 1 | context | |
| `c_L,gk, c_R,gk` | 130:132 | 2 | **goal** | 每手对目标杆 |
| `h_L,gk, h_R,gk` | 132:134 | 2 | **goal** | 每手对目标杆持续 |
| 上一步动作 | 134:148 | 14 | context | 抑制抖动 |
| — | 148:160 | 12 | goal（追加在 obs 末尾）| = 上面 4 个 goal 块的副本 |
| **合计** | | **state 148 + goal 12 = obs 160** | | |

**审计结论（冗余）**：`d(10) + w(10) + c(2) + max(c)(1) + c_next(2)` = **25 维全部是
`p(6)` 加常量杆几何的确定性函数**——同一个 6 维量被编码了 4 遍（d 是原量，w 是 softmax(d)，
c 是 exp(-(dmin/τ)²)，max(c) 是 max(c)）。这是计划当年有意为之（给策略不同尺度的视角），
但既然要重新梳理：最安全的瘦身对象是 `w(10)`（`bar_L/bar_R/k_ref` 是用 `self._grasp()` 现算的，
不读这个特征块），去掉它 state 148 → 138；`d` 也写着"去掉 d_hk + prev_action 可降到 117"
（计划 §3 注）。另外 14 个指关节只在 `q[7:]` 里，**接触特征完全不看它们**（§9.51 的偏离）。

#### 9.54.2 重新梳理后的 g（用户提议，已实现为 `dual6c`）

```
dual6c  g = [x_torso, z_torso, c_L,gk, c_R,gk, h_L,gk, h_R,gk]        6 维, state 145 / obs 151
```

`p_L/p_R(6)` **移出 goal、留在 state**（6 个世界坐标与 `c_*,gk` 说的是同一件事，而后者增益高得多）。
指令仍是单一下标 `gk`（`info["k_goal"]`），因此**目标杆可以每 episode 采样**，`c_*,gk`/`h_*,gk`
自动跟着变。

**一个必须一起测的对照 `dual6d`**（同样 6 维，只把接触编码从饱和换成线性）：

```
dual6d  g = [x_torso, z_torso, d_L,gk, d_R,gk, h_L,gk, h_R,gk]        6 维, state 145 / obs 151
```

理由（两条，都有出处）：
1. **`c` 在远处没有梯度**：`c = exp(-(d/0.04)²)`，d = 0.37 m 时 `exp(-85) ≈ 0` 且梯度 ≈ 0
   ⇒ 目标杆很远时，4 个接触维**给不出"往哪伸手"的方向**；线性距离在任意距离都有梯度。
   代码里 `_advance_features` 的注释早就写了这件事（"per-hand DISTANCE ... not the saturating
   soft contact ... a distance decreases monotonically as a hand reaches out"）。
2. 但它**不能替代** `c`：run #18（把高增益的 `c` 删掉）直接崩成松手坠落（§9.50/§9.51）。

所以 `dual6c` vs `dual6d` 是**同一维数、只差接触编码**的最干净单变量 A/B，同时也回答了
§9.50/§9.51 那个悬而未决的机制问题：起作用的是"指数的高增益"，还是"有一个以目标杆为参照的每手项"。

CPU 实测（t=25，吊在 B0）：

| 变体 | goal | 吊 B0 / goal B1 | 吊 B0 / goal B0（已到达）| 目标值 |
|---|---|---|---|---|
| `dual6c` | 6 | **2.028** | **0.719** | `c=1,1, h=1,1` |
| `dual6d` | 6 | **1.544** | **0.131** | `d=0,0, h=1,1` |
| （对照）`dual_hnext` 12 维 | 12 | 2.093 | 0.721 | |

两个直接后果：

* **阈值**：`dual6d` 在"已吊在目标杆下"只有 **0.131** ⇒ **默认 `--goal-reach-thresh 0.35` 就能用**
  （错杆 1.544，分离 12×，可读性最好）。`dual6c` 因为 c 的固有缺口是 **0.719**，而**任何"不在目标杆上"
  的状态都 ≥ 2.0**（起始杆悬挂 2.028，单手在目标杆上 ≈2.0）⇒ **用 `--goal-reach-thresh 1.0`**：
  它比 0.8 更能容忍抓握略松（0.8 要求 c ≳ 0.45，1.0 只要求 c ≳ 0.30），而离"不在目标杆"仍有 2× 余量。
  （阈值只影响 `success`/`dwell_success` 这两个**读数**，不影响训练——CRL 是无奖励的。）
* **梯度**：`dual6d` 在远距离给出 0.37/0.36 的线性缺口 ⇒ 有方向；`dual6c` 在远距离是 1.0/1.0 的
  饱和缺口 ⇒ 没有方向，只能靠 `x,z`（躯干）和重标记课程引导。

#### 9.54.3 多杆：不需要改代码，直接采样目标杆

用户的新期望是"目标杆多变时，策略自己学会过多杆"。这套机制**已经就位**：`gk` 每 episode 采样，
`c_*,gk`/`h_*,gk` 跟着变，`x,z` 的 goal 取自 `goal_set[gk]`（绝对值 ⇒ 带杆身份）。所以
**单变量**实验就是：**起点固定 B0 + 采样目标杆**（`--train-goal-bar -1 --train-goal-bar-min 1`），
这与 run #16 的失败设置（**同时**随机起点 + 多目标）只差"起点是否随机"这一个变量。

`dual6c` / `dual6d` / `dual_hnext` 都支持（`k_goal` 是共享机制）。

#### 9.54.4 两条实测出来的工程约束（避免再踩）

1. **jaxgcrl 的整除约束**：`crl.py:check_config` 要求
   `num_envs * (episode_length - 1) % batch_size == 0`。本环境 `episode_length = 501`
   ⇒ 128 envs 时 `128*500 = 64000`，所以 `batch_size` 必须整除 64000（512 ✓，与 #13/#17 同）。
   小规模复现要**保持同一比例**：8 envs ⇒ `batch_size = 32`（或 40/50/64/100/125/128/160…），
   否则启动即断言失败（我第一次 CPU 冒烟就踩了这个：8/64 不整除）。
2. **小规模下 `critic_loss=nan` 是已知现象**（§9.34.5，与 goal 变体无关）：CPU 冒烟用
   8 envs / batch 32 / unroll 10 时同样出现，但**配置、环境构建、评估、checkpoint 全流程正常**
   （`[goal] variant=dual6c goal_size=6 state_dim=145 obs=151 indices=(0,2,127,128,129,130)`，
   eval 打出 `episode_success/episode_dist/episode_max_bar`，`actor_000006800.pkl` 落盘）。

### 9.55 结果：`dual6c`（去掉 `p(6)`）失败 —— 与 #18（去掉 `c`）是镜像式的同一种崩塌

`runs/brach_dual6c_b1/`（20 evals / 1.22e7 步 / 4770 s），`--goal-reach-thresh 1.0`，
goal = `[x, z, c_L,gk, c_R,gk, h_L,gk, h_R,gk]`（6 维）。**20 次 eval `success` 恒 0**，
`dwell_success` 恒 0。与 run #13 同预算对照：

| 指标 | #13 `support_dual`（11 维）| **#19 `dual6c`（6 维）** | #18 `dual_nomax`（10 维）|
|---|---|---|---|
| `d/len`（末）| 0.830 | **1.741**（= 它的"不在目标杆"水平 2.0）| 1.389 |
| `avg_episode_length` | 473.8 | **145.5** | 128.2 |
| `cov_dual_runmax`（最长同杆双持）| 69.8 | **2.9** | 9.8 |
| `hand_on_steps / len` | 371/474 = **0.78** | **75.7/145 = 0.52** | 60/128 = 0.47 |
| `bar_L` / `bar_R`（均值）| +0.79 / −0.08 | **+0.11 / −0.77** | −0.77 / −0.37 |
| `fell`（末）| 6% | **100%** | **100%** |
| `success`（末）| 152.2 | **0** | **0** |
| `training/categorical_accuracy`（eval 1 → 末）| 0.354 → 0.185 | **0.011 → 0.140** | 0.358 → 0.263 |

**它"够得到，但抓不住"**：`cov_cross_b1_010 = 1.0`、`cov_min_d_b1_improve = 1.995`
⇒ 手最近到了离 B1 **5 mm** 的地方（`max_bar` 也因此有 130/145 ≈ 0.9），但
`cov_b1_runmax = 0`（从未在 B1 的窗内停留）、`dualMx` 只有 2.9 步、`bar_R` 均值 −0.77、
`fell` 100% —— **一集 145 步就掉下去**。

#### 9.55.1 结论：`p(6)` 与高增益接触项**缺一不可**（本轮最重要的结果）

| run | 目标里有什么 | 结果 |
|---|---|---|
| #13 / #17 `support_dual` | `x,z + p(6) + max(c) + h_any + h_dual` | **成功**（双持 425 步、fell 0%）|
| #18 `dual_nomax` | `x,z + p(6) + h_any + h_dual`（**去掉 c**）| 崩塌：fell 100%、succ 0 |
| **#19 `dual6c`** | `x,z + c_*,gk + h_*,gk`（**去掉 p(6)**）| **崩塌：fell 100%、succ 0** |

⇒ §9.54 里"把 `p(6)` 移出 goal"的建议**被证伪**。两者是**互补角色**，不是二选一：

* `p(6)`（世界坐标，米制增益 ≈ 1/m）提供**任意距离上的方向**。
  ⚠️ **下面这段「goal 退化」的解释已被 §9.55.3 的实测更正**：末期策略的接触维并非恒 0，真正的原因是「缺席那只手在远处零梯度」。原文保留以见过程：
  它同时保证**重标记 goal 的多样性**：策略崩坏时身体仍在空间里移动 ⇒ 未来状态的 `p` 仍然千差万别，
  对比任务有信号；而 `c`/`h` 在"没抓住"时**齐刷刷都是 0**，goal 会退化成几乎同一个向量。
  `categorical_accuracy` 在 eval 1 是 **0.011**（#13 同期 0.354）正是这个退化的直接读数。
* `c`（`exp(-(d/0.04)²)`，杆附近增益 ≈ 21/m）提供**让"抓住"本身值得做**的量级（§9.50/§9.51）。

⚠️ 另有一个不能排除的候选机制：goal 换成 6 维后，**初始（随机）策略的行为就变了**——
#13 的 eval 1 是"单手吊满整集"（`single_on_steps = 493`），而 `dual6c` 的 eval 1 是
"全程几乎不抓"（`hand_on_steps = 14/501`）。也就是说它可能一开始就被推进了"松手"吸引子，
之后再也爬不出来（两种崩塌 run 的末期曲线形状一致：`hand_on` ≈ 半个集、`fell` 100%）。

#### 9.55.2 由此调整下一步（`dual6d` 不再值得跑）

`dual6d` = `dual6c` 把 `c` 换成线性距离，**同样没有 `p(6)`** ⇒ 按上面的结论，预期会以同样方式失败，
跑它属于低价值。真正要回答的问题是**"#13 的两个改动各自是否安全"**，而这两个变体都保留 `p(6)`：

| 变体 | 相对 #13 只改了什么 | 维数 | 预期问题 |
|---|---|---|---|
| `dual_cnext` | `max(c_L,c_R)`（任一手/任一杆）→ **每手对目标杆**的 `c_L,gk,c_R,gk` | 12 | 保住承重项；消除"任意杆"退化 |
| `dual_hnext` | 再把 `h_any/h_dual` → **每手对目标杆**的 `h_L,gk,h_R,gk` | 12 | 连"等着 h 长大"和"错杆白送"一起堵掉 |

两者与 #13 **同维级、同配方、同预算**（20 evals / 1.22e7 步），是当前最有信息量的两次。建议
**串行**跑（这台 dual4090 并发会崩，见 `docs/复现环境.md` §4.7）：
先 `dual_hnext`（一次把两个改动都换掉），若它成功但与 #13 有差，再用 `dual_cnext` 分离出是哪一个改动带来的。

#### 9.55.3 更正（实测）：不是"goal 退化"，而是"缺席那只手没有梯度"→ 卡死在单手吸引子

上面 9.55.1 里我把机制写成"`c`/`h` 齐刷刷都是 0 ⇒ goal 退化成几乎同一个向量"。**这句话不准确，
实测部分推翻了它**。用 `.scratch/goal_spread.py` 回放 `runs/ckpt_dual6c_b1`（在第一次 `done` 处截断，
只看存活段），逐维统计 goal 各维的幅度（min/mean/max/range）：

| goal 维 | **末期 ckpt**（12.2M 步，存活 216 步）| **早期 ckpt**（738k 步，存活 501 步）|
|---|---|---|
| `x` | −0.007 / 0.305 / 0.563（range **0.571**）| −0.009 / 0.082 / 0.132（0.140）|
| `z` | 0.684 / 0.789 / 0.900（0.216）| 0.772 / 0.931 / 0.964（0.192）|
| `c_L,gk` | 0.000 / 0.482 / **0.976**（range **0.976**）| **恒 0** |
| `h_L,gk` | 0.000 / 0.488 / **0.993**（range **0.993**）| **恒 0** |
| `c_R,gk` | **恒 0**（range **0.000**）| 恒 0 |
| `h_R,gk` | **恒 0**（range **0.000**）| 恒 0 |
| `dist` | 2.029 → 2.002（mean 1.663）| 恒 2.022 |
| `max_bar` | 1.0 | 0.0 |

**读出来的事实**：

1. **末期策略是"单手抓住目标杆"**：左手 `c_L,gk` 冲到 0.976（≈0.7 cm，真的贴上了 B1）、
   `h_L,gk` 冲到 0.993（在 B1 窗内连续约 50 步），**右手四个维度全程恒 0**。
   所以 **goal 的接触维并非全部常值**，`c`/`h` 确实带着对比信号——我的"退化"论断对末期不成立。
2. 但 **`dist` 依然停在 2.0**：因为 goal 是**每只手各自**要求（`c_R,gk`、`h_R,gk` 各有 1.0 的缺口）
   ⇒ 只要右手不上目标杆，`dist ≥ sqrt(1²+1²) = 1.414`，**永远高于阈值 1.0**。
   也就是说 **`success` 恒 0 是被"单手"这个状态结构性地保证的**，不是"没走到附近"。
3. **早期 ckpt 才是"全维恒 0"的那种情形**：它整集 501 步吊在 B0、`max_bar`=0、四个接触维恒 0、
   `dist` 恒 2.022 ⇒ 这一阶段确实所有接触维都不带信号（与 eval 1 的 `categorical_accuracy=0.011` 一致）。
4. **真正的原因（与 `dual6d` 那次我提出的饱和问题同一个）**：`c = exp(-(d/0.04)²)` 在 d = 0.35 m 时
   是 5.6e-34，梯度 `dc/dd = -2 d c / tau^2` ≈ **-2.5e-31（float32 里就是 0）**。
   于是**右手那两维在远处是"恒 0 且零梯度"**——它给不了"把右手伸向 B1"的任何方向。
   而 `p(6)`（米制世界坐标）正是补这一块的：它在任意距离都有 ≈1/m 的线性梯度，会把**两只手都**拉向目标杆。
   这就把三种失败串成一条线：

   | run | 有 `p(6)`？ | 有高增益 `c`？ | 结果 |
   |---|---|---|---|
   | #13/#17 `support_dual` | ✅（两手都有**到达**梯度）| ✅ | 成功，双持 425 步 |
   | #18 `dual_nomax` | ✅ | ❌ | 两只手都够得着，但**没有"抓住"的理由** ⇒ 松手坠落 |
   | **#19 `dual6c`** | ❌ | ✅（但在远处零梯度）| 先伸到的那只手抓住 B1，**另一只手永远没有梯度** ⇒ 卡在单手吸引子，`success` 结构性为 0 |

   ⇒ 9.55.1 的结论"两者缺一不可"仍然成立，但机制要说成：
   **`p(6)` 提供"两只手都能到达"的梯度（并让早期的重标记 goal 有差异），`c` 提供"抓住"的量级，
   `h` 提供持续性**；三者各司其职，删掉 `p` 的直接后果不是"goal 退化"而是**缺席那只手失去唯一的方向来源**。#### 9.55.4 渲染确认（`runs/render/render_dual6c_final`，末期 ckpt，goal_bar=1，2 集 × 260 步）

用户问「dual6c 其实抓住了 B1？」——**是的，但只有一只手**：

| | ep0（done@115）| ep1（done@116）|
|---|---|---|
| 左手在 B1 抓握窗内 | t=17–45 / 48–61 / 65–80（**59 步，最长连续 29**）| t=34–37 / 42–49（12 步）|
| 两手各一根杆（左 B1 + 右 B0）| 22 步 | 8 步 |
| 右手在 B0 窗内 | t=0–20 / 25–35 / 42–45 | t=0–18 / 28–35 / 42–55 |
| 离 B1 最近距离 | 左 **0.003 m** @t=73；右 0.258 m @t=48 | 左 0.019 m；右 0.234 m |
| 右手是否进过 B1 窗 | **从未** | **从未** |

时间线（ep0）：`t=0` 双手吊 B0 → `t≈5` **左手松开**（右手独撑）→ `t≈17–20` 左手够到 B1
（`d_LB1` 降到 3 cm，`bar_L`=1），**右手仍在 B0** ⇒ 进入「左手 B1 / 右手 B0」的**两杆分腿支撑** →
维持到 `t≈80`（其间三次短暂滑出）→ `t≈85` 左手打滑（`d_LB1` 0.02→0.09）→ **`t=115` 坠落**。
ep1 更差：先向后荡（`qpos_x` 到 −0.18）再掉。

**所以**：策略确实学会了「**单手完成转移并抓住下一根杆**」——这正是设计里要的中间态
（§9.51.4 的 `(c_L=1, c_R=0)` 结构）——但它**从不把第二只手带过去**（右手最近 23 cm，
`dualMx`≈2.9 步），而 goal 是**每只手各自**要求 ⇒ `dist ≥ sqrt(1²+1²) = 1.414 > 阈值 1.0`
⇒ `success` 结构性为 0；分腿支撑维持不住，最终打滑坠落（`fell` 100%）。
与 §9.55.3 的结论一致：**缺席那只手没有梯度**，所以「只差把右手也带过去」这一步它自己走不到——
而补这个梯度正是 `p(6)` 的作用，所以下一步该跑保留 `p(6)` 的 `dual_cnext` / `dual_hnext`。

复现：`JAX_PLATFORMS=cpu MUJOCO_GL=egl .venv-warp/bin/python src/render_policy.py \
  --ckpt runs/ckpt_dual6c_b1/actor_latest.pkl --goal-bar 1 --episodes 2 --steps 260 \
  --lookat-x 0.25 --distance 2.6 --name render_dual6c_final`
（产物：`runs/render/render_dual6c_final/{render_dual6c_final.gif, frames/step_%04d.png, trace.csv}`；
关键帧：`0000` 双手 B0、`0015` 左手松开右手独撑、`0030` 左右手各一根杆、`0080` 即将打滑、`0100` 坠落。）### 9.56 `dual6d` 也失败（第三种失败形态）+ 一次「静默跑 CPU」事故与守卫

`runs/brach_dual6d_b1`：20 evals / 1.22e7 步 / **6028 s**（1.68 h，跑在**本地单卡机 `Soil` 的 `--gpu 0`** 上）。
`--goal-reach-thresh 0.35`（默认），goal = `[x, z, d_L,gk, d_R,gk, h_L,gk, h_R,gk]`（6 维）。

| 指标（末 eval）| #13 `support_dual` | #19 `dual6c` | **#20 `dual6d`** |
|---|---|---|---|
| `succ` | 152.2 | 0 | **0** |
| `advance_max` | 391.7 | 96.1（末）| **0.0（全 20 个 eval 恒 0）** |
| `max_bar`（均值=杆号）| 0.97 | 0.90 | **0.18**（峰值 25.8，多数 episode 为 0）|
| 离 B1 最近距离 | 0.002 m | **0.005 m** | **0.125 m**（末 eval 0.20 m）|
| `dual_runmax` | 69.8 | 2.9 | 12.6（eval 18 曾到 69.8）|
| `hand_on_steps/len` | 0.78 | 0.52 | 0.34 |
| `fell`（末）| 6% | 100% | **100%** |
| `categorical_accuracy`（末）| 0.185 | 0.140 | 0.116 |

渲染（`runs/render/render_dual6d_final`，末期 ckpt，goal_bar=1）：`episode max bar = 0`
（**从未离开 B0**）、`fell at [136, 149]`、离 B1 最近 **0.212 m**、
**`grasping (soft) = L 0.15 R 0.12`**（对照 #17 是 0.88/0.87）、`hand_switches` 17/20（反复重抓）。

⇒ 三种失败形态互不相同，但都指向同一件事：

* `dual6c`（有 `c`/`h`、无 `p`）：**单手够到并短暂抓住 B1**（5 mm），另一只手没有梯度 ⇒ 卡在单手；
* `dual6d`（线性 `d`/`h`、无 `p`、无 `c`）：**根本不尝试转移**（`advance_max` 恒 0），
  手滑着挂在 B0 上反复重抓（soft grasp 仅 0.15）⇒ 最终坠落；
* `dual_nomax`（有 `p`、无 `c`）：够得着但**没有"抓住"的理由** ⇒ 松手坠落。

⇒ **"把 `p(6)` 移出 goal" 这条线到此为止**（`dual6c`、`dual6d` 都失败，`dual6d` 已标为证伪）。
下一步只跑**保留 `p(6)`**、只改接触项参照方式的 `dual_cnext` / `dual_hnext`（12 维，§9.55.2）。

#### 9.56.1 事故：`--gpu 1` 在单卡机上导致「静默跑 CPU」

本地机 `firedust@Soil` 上跑 `--gpu 1` 时：`--gpu 1` 会设 `CUDA_VISIBLE_DEVICES=1`，
而该机**只有一张卡（索引 0）** ⇒ 把唯一的 GPU 藏起来 ⇒
`cuInit(0) failed: CUDA_ERROR_NO_DEVICE` ⇒ JAX **只打印一行 warning 就回退 CPU 并开始训练**。
日志里唯一的判据是：

```
[gpu] CUDA_VISIBLE_DEVICES=1 jax.devices()=[CpuDevice(id=0)]
```

它没有崩，所以不注意就会在 CPU 上白跑（那次是 `^C` 手动停的）。
反证：同一台机器把 `--gpu` 换成 **0** 后，`dual6d` 正常跑完（1.68 h）⇒ **GPU 与驱动本身没问题**。

**据此给 `train.py` 加了硬性守卫**（提交见本次改动）：JAX 后端不是 GPU、且**没有显式声明** CPU
（`JAX_PLATFORMS` 不含 `cpu`）、也没有 `--allow-cpu` 时，直接 **`SystemExit(3)`**，并打印三类常见原因
（`--gpu` 索引不存在、驱动/设备不可见、`JAX_PLATFORMS=cpu` 被别的 shell 继承）。两条路径都实测过：

* `JAX_PLATFORMS=cpu` + 同样的配置 ⇒ 放行（打印 `[gpu] CUDA_VISIBLE_DEVICES=<unset> jax.devices()=[CpuDevice(id=0)]`）；
* 未声明 CPU + `--gpu 0`（驱动不可见时）⇒ `[gpu] FATAL: ... refusing to train on CPU.` 且**退出码 3**。

⇒ 以后 GPU 运行前的判据不变（看 `[gpu]` 那行的 `CudaDevice`），但现在**忘了看也会被拦下来**。### 9.57 为什么 `dual6d` 只"挺下半身"、双手不肯尝试 —— CRL 自课程的信道问题

用户看渲染发现：**机器人努力把下半身向 B1 挺，手却抓着 B0 不动**。实测 `dual6d` 末期策略 rollout
（148 步存活，`max_bar`=0）各 goal 维幅度：

| goal 维 | min | mean | max | range |
|---|---|---|---|---|
| `x_torso` | −0.181 | 0.040 | **0.426** | **0.607** |
| `z_torso` | 0.667 | 0.861 | 1.026 | 0.359 |
| `d_L,gk` | 0.294 | 0.383 | 0.534 | 0.240 |
| `d_R,gk` | **0.203** | 0.377 | 0.702 | 0.500 |
| `h_L,gk` / `h_R,gk` | 0 | 0 | 0 | **0.000** |

**机制（三条，合起来正好解释"挺下半身"）**：

1. **CRL 训练时的 goal 是"同一轨迹的未来状态"重标记**（`goal = future_obs[:, goal_indices]`，γ^Δt 加权）。
   因此 **策略从没做过的行为，永远不会被当成目标**：既然两只手从没到过 B1，未来状态里就没有
   "手在 B1"，于是**没有任何一维要求它把手送过去**。行为只能先被探索撞上、再被强化。
2. **只有"在未来状态里会变"的维才带学习信号**。这里 `x_torso`（range 0.61）与 `z_torso`（0.36）
   在探索中天然大幅变化 ⇒ 梯度几乎全落在躯干上 ⇒ 策略学会把骨盆/腿往 B1 方向送
   （实测 `x` 最大到 **0.426**，已经越过 B1 的 0.335）——这正是用户看到的"挺下半身"。
3. **终止条件那一维是死的**：`h_*,gk` 要求"某只手连续 30+ 步在 B1 窗内"，而它**在所有未来状态里恒为 0**
   ⇒ 对对比任务**零信息、零梯度**。`d_*,gk` 虽然会变（range 0.24/0.50），但它的 goal 取自**自己的未来**
   （≈0.35，与现状相同）⇒ 目标是"保持 0.35"，**没有把手往 0 推的压力**（只有 eval 用的指令 goal 才是 0）。

**这也是为什么 #13 能成、#18 会崩**（假设，但与所有测量一致）：

* #13 的 `max(c_L,c_R)` 是**"吊着不动就能改善"的阶梯**——拉紧抓握（把抓握座从 3.2 cm 收到 ~0.7 cm）
  就把它从 0.52 推到 ~0.97，探索很容易撞上 ⇒ 被强化 ⇒ 而"往上拉紧"在力学上正是摆荡/转移的前置动作
  ⇒ 于是自然滚到"一只手荡到 B1"。
* #18 删掉它之后，**唯一容易撞上的新行为是"松手"**（松手会让 `p(6)` 大幅变化）⇒ 被强化 ⇒ 崩成坠落。
* `dual6c`：左手碰巧荡进 B1 附近（`c_L,gk` 从 0 跳到 ~1，出现大幅对比差异）⇒ 被强化；
  右手从没碰巧到过（且 `c` 在 0.35 m 处梯度为 0，漂不过去）⇒ 永远 0。
* `dual6d`：`d` 是线性的、有梯度，但**目标值取自自己的未来**（≈现状）⇒ 没有"再靠近一点"的压力；
  `h` 恒 0 没有梯度 ⇒ 只有躯干维在动。

**设计结论（比 §9.55 更具体）**：goal 里必须有一个**"阶梯"维**——它满足
(i) **吊着不动就能被改善**（探索容易撞上、且改善方向与最终目标机械相关），
(ii) 它的**目标值是"更好"而不是"和现在一样"**（指令 goal 恒定，而不是靠未来状态重标记）。
`max(c_L,c_R)`（抓握质量）恰好满足这两条；`c_*,gk`（对目标杆的接触）不满足 (i)；
线性 `d_*,gk` 不满足 (ii)；`h_*,gk` 两条都不满足（在到达之前恒 0）。

⇒ 因此**最稳的下一步不是把 `h_any/h_dual` 换成 `h_*,gk`（那会同时删掉 `c_any` 这条阶梯）**，
而是：**保留 `c_any`（阶梯）+ `p(6)`（到达几何）+ 用 `h_*,gk` 作终止条件**，即
`g = [x, z, p_L(3), p_R(3), c_L, c_R, h_L,gk, h_R,gk]`（12 维，记为 `dual_ladder`）。
它相对 #13 只改"持续性判据的参照物"（`h_any/h_dual` → 每手对目标杆），**保住了唯一被证明能自举的阶梯**。### 9.58 「只用 x,z,p(6) 行不行」—— 我们早就跑过，实测阶梯如下

用户问：能不能只用 `x, z, p_L(3), p_R(3)`（即 `position`，8 维），以及与加上 `c`、`h` 的区别。
**答案全在已有实验里**（run #3 的 `args.json` 里 `goal_position_only: 1`）：

| run | goal 里有什么 | `len`（末）| `fell`（末）| `succ`（末）| `max_bar`（末）| `d/len`（末）| 结论 |
|---|---|---|---|---|---|---|---|
| **#3** `brach_pos8_b1` | `x, z, p(6)`（8 维）| **501** | **0%** | **0** | **0.0（一整轮恒 0）** | 0.62 | **稳定吊在起始杆，从不移动** |
| #10 `brach_d035_reach` | `+ max(c_L,c_R)` | 40.9 | 100% | 7.1 | 12.1 | 0.67 | 会去够，但**飞掠即达标**（"双手离杆鱼跃"，最好一步 dist=0.06）|
| #11 `brach_hold_b1` | `+ h_any` | 306 | 44% | 108 | 289 | 0.57 | **碰到变成持续抓住** |
| #13 `brach_dual_b1` | `+ h_dual` | 474 | 6% | 152 | 459 | 0.83 | 单手侧吸引子被打破（双持 69.8 步）|
| #17（6h）| 同 #13，预算 4× | 501 | **0%** | 416 | 470 | **0.198** | 双持 425 步、两只手都在 B1 |

（#3 跑的是更容易的 `full`（0.40 m）**且用了 40 evals = 双倍预算**，所以"不动"不是预算不够；
#10/#11/#13 都在 `full035` 上，彼此可比。`d/len`：`position` 在"仍吊 B0、目标 B1"时正是 **0.62**
——即它连一次"靠近"都没有发生过。）

**三者的分工（各管一件事）**：

* `p(6)`：**能不能到**。3 维带符号误差、任意距离都有 ≈1/m 的方向 ⇒ 到达靠它；
  但它**不区分"抓住"与"飞过/悬在旁边"**，而且（§9.57）它的重标记目标取自自己的未来 ≈ 现状
  ⇒ 单靠它**没有"往目标推"的压力** ⇒ #3 那种"稳定吊着不动"。
* `c`：**愿不愿意抓**（也是唯一被证明能自举的**阶梯**）。高增益（杆附近 ≈21/m），而且
  **吊着不动就能改善**（把抓握座从 3.2 cm 收到 0.7 cm，0.52 → 0.97），探索容易撞上、方向又与
  "荡过去"机械相关 ⇒ 把"不动"变成"会去够"（#3 → #10）。
* `h`：**抓住了算不算数**（持续性）。把瞬时飞掠排除 ⇒ 把"飞掠"变成"抓住"（#10 → #11）。

⇒ 因此：**不要只用 `x,z,p(6)`**（#3 证明它不动）；**`c` 要留**（阶梯）；真正值得改的是
**`h` 的参照方式**（`h_any`/`h_dual` 会被起始杆白送，§9.52）——即 §9.57 提的
`dual_ladder = [x, z, p(6), c_L, c_R, h_L,gk, h_R,gk]`（12 维，只把持续性判据换成"每手对目标杆"）。### 9.59 `max(c_L,c_R)` 会不会鼓励"赖在 B0 上"？（用户提问）

**不会——它既不奖励 B0，也不能把"原地"变成解。** 分四层：

1. **它是"杆无关"的**：`c_h = exp(-(dmin_h/0.04)^2)`，`dmin_h` 是这只手到**最近一根杆**的距离
   ⇒ 抓紧 B0 得 c≈1，抓紧 B1 也 c≈1。它说的是"**稳稳坐在某根杆上**"，不是"坐在 B0 上"。
2. **"原地 + 拉紧"够不到目标**（用 §9.51 实测的逐维分解算）：吊在 B0、目标 B1 时几何项平方和
   = **0.3822**（这部分只有真的移动才能消掉）。即使原地把抓握质量从 0.53 拉到 1.00，
   `|diff|` 也只从 **0.789 → 0.629**，而阈值是 **0.35** ⇒ **它不可能成为与"转移"竞争的另一个最优解**。
3. **`max` 这个形式恰恰是"允许松一只手"的关键**（计划原本的设计意图：
   "松一只手不受罚、丢最后一只手受罚"）：松掉左手时 `c_L` 掉到 0，但 `max(c_L,c_R)` 由仍抓着 B0 的
   右手维持住 ⇒ **松开一只手是免费的**；只有"两只手都离杆"才被重罚（这正是 #18 崩成坠落时缺的约束）。
   如果换成 goal 里的**每手** `c_L,c_R`，松左手会立刻让该维归零 ⇒ **转移的第一个动作被惩罚**，
   这就是 `full` 变体（"要求两只手分别有接触"）当年不好用的原因。
4. **实测上它与"赖着不动"是反相关的**：run #3（只有 `x,z,p(6)`、**没有 c**）整轮 `max_bar` 恒 0、
   从不移动（虽然 `fell` 0%、能稳定吊满整集）；run #10（**加上 `max(c)`**）立刻开始够
   （`max_bar` 6.8→12.1）。也就是说**加上 c 才是"赖着不动"这个最优解被打破的原因**。

**诚实的边界**：上面 1–3 是定义与算术，4 是实测；但"c 为什么能带来前进"仍**是假设**，两种读法都与证据相容：

* **阶梯说**（§9.57）：它是唯一"不离开起始杆也能改善"的维（0.52→0.97），而改善它的动作（往上拉紧/摆荡）
  在力学上正是转移的前置；
* **保险说**：没有它时，探索唯一容易撞上、又能让任何 goal 维变化的新行为就是"松手"（`p` 大幅变化）
  ⇒ 被强化 ⇒ 崩（#18）。

两种读法都认为 c 是承重的，只是路径不同。**判别实验（单变量，两者都保留 `p(6)` 和目标杆持续量）**：

| 变体 | 接触维 | 能否在 B0 上被满足 |
|---|---|---|
| **A `dual_ladmax`** = `[x,z,p(6), max(c), h_L,gk,h_R,gk]`（11 维）| 任意杆的 `max(c)` | **能**（B0 上 0.52，拉紧到 ~0.97）|
| **C `dual_hnext`** = `[x,z,p(6), c_L,gk,c_R,gk, h_L,gk,h_R,gk]`（12 维，已实现）| 对目标杆的每手 `c` | **不能**（B0 上 5.6e-34）|

A ≫ C ⇒ "能在 B0 上被满足"这件事本身是承重的（用户的怀疑在这一意义上成立）；
A ≈ C ⇒ 接触项的参照物无关，c 只是普通的抓握质量项。### 9.60 更正：#18 不是"松手坠落"，而是"抓住后打滑"——`c` 的作用要改写

用户看渲染提出：#18（`dual_nomax`，无 `c`）最后是**一只手拽着 B1、另一只手在往 B1/B2 够**，
而且"加大训练时间可能能成"。按存活段重测 `runs/render/render_check_nomax`：

| | ep0 | ep1 | [对照] 含坠落的全 501 步 |
|---|---|---|---|
| 存活段长度 | **128 步**（`done@128`）| 128 步 | 501 |
| 有任一只手在杆上的步数 | **90/128 = 70%** | 91/128 = 71% | —— |
| 左手在 B1 窗内 | **10 步（t=113–125）** | 11 步（t=113–125）| —— |
| 右手在 B1 窗内 | 0 步（最近 **0.169 m**）| 0 步（最近 0.169 m）| —— |
| 两手都不在杆上 | 38 步 | 37 步 | **412/501 = 82%（我上次引用的就是这个数）** |

**时间线（ep0）**：t=0 双手吊 B0 → t≈20–110 **大幅摆荡/腾空**（`qpos_x` 走到 +0.43、z 到 0.876，
两手常同时离杆）→ **t=113 左手抓住 B1**（`d_LB1`=0.029，躯干 x=0.426 已越过 B1 的 0.406）→
t=113–125 **吊在 B1 上约 13 步**（其间 `d_LB1` 在 1.4–5.4 cm 之间晃，两次跌出 5 cm 窗）→
**t=126 左手打滑脱开（`d_LB1` 0.08→0.13→0.17）→ t=128 坠落**。右手全程在够，最近 0.169 m，从未进窗。

⇒ **更正两处**：

1. **§9.51 的"82% 时间两手都不在杆上 ⇒ 松手坠落"是错的**（把 `done` 之后的自由落体算进去了；
   存活段是 70% 有手在杆上）。#18 的真实失败是「**鱼跃起跳 → 单手抓住 B1 → 荡回时握不住、滑脱 → 掉**」，
   和 #19（`dual6c`，单手抓住但另一只手没有梯度）属于**同一类**"单手到位、守不住"的失败，
   不是"学成松手"。
2. **`c` 的作用要改写**：它主要不是"阻止探索去松手"（§9.57 的"保险说"表述撤回），
   而是**"到位后不让抓握滑脱"**——它把抓握座从"晃在 1.4–5.4 cm、随时跌出窗"拉回到 0.7 cm：
   杆附近 `dc/dd = -2dc/tau^2 ≈ -25 /m`，是目标里唯一能提供这种"咬住"量级的项。
   实测对照：#18 的左手在 B1 上时 `d_LB1` 在 1.4–5.4 cm 之间晃（`c` 0.16–0.89、多次跌出窗），
   而 #17 的稳态抓握座离杆心 ~0.7 cm（`c` 0.96–0.98）——**这就是"抓住"和"抓住但滑"的区别**。

这也让整条阶梯更自洽（每条 run 只差一个东西）：

| run | goal | 结果 |
|---|---|---|
| #3 | `x,z,p(6)` | **完全不尝试**（`max_bar` 恒 0）|
| #18 | `+ h_any,h_dual` | **会鱼跃、单手抓住 B1，但守不住**（抓 13 步后滑脱；`succ` 0）|
| #10 | `+ max(c)`（无 h）| 会够、瞬时达标（`succ` 7.1、`max_bar` 12.1）但 100% 掉 |
| #11 | `+ max(c) + h_any` | **抓住并守住**（`succ` 108、`fell` 44%、`len` 306）|
| #13/#17 | `+ h_dual`；4× 预算 | 双持 425 步、`fell` 0% |

**关于"加大训练时间也许能成"**：不排除，而且这个渲染支持这个方向——#18 已经完成了"起跳+单手抓住"，
差的只是"守住的最后一步"（而它在 eval 10 也曾短暂恢复到 `len` 409）。要与"改 goal"区分开，
需要一次**单变量**的长跑：把 #18 的原配方跑满 6 h（60 evals），看"守不住"是否只是时间问题。#### 9.60.1 #18 的逐 eval 趋势：**接近在改善，守住没有出现**（"加大训练时间"该不该赌）

`runs/brach_dual_nomax_b1/progress.csv` 全程 20 个 eval（`mind` = 离 B1 最近距离，由 `cov_min_d_b1_improve` 反解）：

| 阶段 | eval | `len` | `dualMx` | `mind`(m) | `max_bar` | `succ` | `fell` |
|---|---|---|---|---|---|---|---|
| 早期 | 1–3 | **501** | 490→10 | 0.24–0.30 | 0 | 0 | 0 |
| 中期 | 4–8 | 68–116 | 4–38 | 0.094–0.239 | 0–0.6 | 0 | 0.94–1.0 |
| 波动 | 9–13 | 153–501 | 5–73 | 0.074–0.284 | 0–1.4 | 0 | 0.19–1.0 |
| **末期** | 14–20 | **122–130** | 3.6–9.8 | **0.022 → 0.015 → 0.031 → 0.016 → 0.018** | 5.9 → **15.8** | **0** | **1.00** |

* **接近端在稳定改善**：最后 4 个 eval 都能到离 B1 **1.5–2 cm**，`max_bar` 也从 5.9 涨到 15.8
  （≈ 有 16 步"最近的杆是 B1"）——与渲染里"左手抓住 B1 约 13 步"一致。
* **守住端没有任何迹象**：`len` 卡在 ~125–130、`fell` 恒 1.00、`succ` 整轮 0（对照 #11 在 eval 20
  已经 `succ` 108、`fell` 44%）。
* 渲染 4/4 集：左手每集都抓住 B1（最近 0.008–0.033 m）、右手稳定差 0.166–0.191 m、全部 t≈128 坠落。

**顺带一个反直觉的对照**：所谓"鱼跃"并不是 `max(c)` 那一族独有的——`air_runmax`（最长腾空段）
#18 ≈ **17.8 步**、#13 **20.1 步**、#17 **10.6 步**，三者同级。**区别不在"跳不跳"，而在"落地后守不守得住"**。

**结论**：这确实是一个"只差最后一步"的失败（不是"没学会"），所以"加大训练时间也许能成"**不能排除**；
但 20 个 eval 里守住端毫无出现迹象，而 #17 是 eval 12–15 才起飞的 ⇒ 最干净的判定就是
**把 #18 的原配方跑满 6 h（60 evals）**：若守住端仍为 0，则 `c` 确实是必需的；若开始守住，
则"去掉 `c` 只是慢"。### 9.61 有/无 `max(c)` 的策略形态差异（用户观察，逐条量化）

用户看渲染指出：**有 `max(c)` 的都像"鱼跃"，去掉的（`dual6c`、#18）不是——而是"一只手继续抓住 B0、
另一只手去够下一根"**。把七条回放的接触签名从轨迹里直接量出来（`runs/render/*/trace.csv`；
先验证过各回放的指令目标都是 B1：t=0 时 `d_LB1=d_RB1=0.351/0.350`，且各自的初始 `dist` 与其 goal 维数吻合）：

| run | 有 max(c) | 存活 | 两手同杆 | **分腿（两手各一根杆）** | 单手 | 零支撑(最长连续) | 首次触 B1 |
|---|---|---|---|---|---|---|---|
| #10 `support` | ✅ | 38 | 21 | **0** | 5 | 6 | t=28 |
| #11 `support_hold` | ✅ | 163 | 53 | **0** | 86 | 18 | t=17 |
| #13 `support_dual` | ✅ | 401 | 135 | **0** | 210 | 32 | t=19 |
| **#17 `support_dual` 6h** | ✅ | 502 | **478** | **0** | 13 | 10 | t=31 |
| #18 `dual_nomax` | ❌ | 128 | 10 | 0 | **79** | 19 | t=114 |
| #19 `dual6c` | ❌ | 115 | 4 | **16** | 42 | 49 | t=17 |
| #20 `dual6d` | ❌ | 142 | 21 | 0 | 76 | 17 | **从未** |

**支持用户观察的两点**：
1. **`分腿`（一只手在 B1、另一只在 B0 的两杆桥）只出现在无 `c` 那一家**（#19 有 16 步，其余全 0）；
   有 `c` 的四个 run 一次都没有过。
2. **"两手同杆"占比**：有 `c` 的（尤其 #17）远高（478/502 = 95%），无 `c` 的只有 10/4/21 步 ⇒
   有 `c` 的最终会**两只手一起落到同一根杆上**，无 `c` 的**始终停在单臂支撑**（单手 79/42/76 步）。

**同时撤回我上一条的推理**：我用 `air_runmax` 得出"三家都有腾空、鱼跃是共有的"——这个代理不成立。
零支撑最长段并不能区分两家（#13 有 32 步、#19 反而 49 步）。**真正的区别不是"跳多久"，而是
"是否出现分腿 / 是否敢于放弃旧支撑去两手一起落"**。

#### 9.61.1 一个几何上的解释（假设，不是结论）

分腿姿势下（左手在 B1、右手在 B0，两杆相距 0.35 m），**后手无法在维持支撑的同时够到 B1**：
手要再走约 0.35 m，而臂展有限；一旦松手又没有摆幅动能、也没有别的支撑 ⇒ 掉。
实测正是如此：#19 维持分腿 16 步后打滑；#18 直到 t=114（存活 128 步）才单手抓住 B1，t=126 就滑脱。

有 `c` 的那一家学到的是一条**"短暂腾空、两手一起落"**的路线：零支撑窗口 6–32 步（0.12–0.64 s），
落地后 `max(c)`（杆附近增益 ≈21/m）让"重新贴住杆"变成一件**回报极高、可以瞬间完成**的事，
而 `h_dual`（两手同杆且持续）正是奖励"两手一起落"的那一维。反过来说，`max(c)` 在腾空期间
是 0（缺口 1.0，目标里最差的状态）⇒ 它同时**惩罚长时间腾空**，所以窗口都很短（≤32 步）。

若这个解释成立，则"锚定式"（一次一只手）在本杆距下是**几何死路**，而不是"还没学会"；
要让它可行，goal 里需要有奖励"后手在维持支撑下逐步靠近目标杆"的项
（`d_*,gk` 那条思路有，但被"目标=自己的未来"抵消了，见 §9.57）。

**判别（单变量、不急着下结论）**：
* 把 #18 的原配方跑满 6 h：若它自然长出"分腿 → 前手换到 B1 → 两手同杆"，说明锚定式只是慢；
  若仍是单臂支撑主导 + `fell` 100%，说明它是死路。
* 同时不要动 goal（避免两个变量一起改）。### 9.63 `dual6c` 跑满 6 h 能不能"双手稳在 B1"？——数据说：**很可能不能**（右手没有梯度）

用户观察："#19 现在能单手抓住 B1，但另一只手似乎在乱转"，问 6 h 是否可能变成双手稳在 B1。

**探针**（`.scratch/rprobe.py`：对每个 checkpoint 跑 6 集、每集 ≤260 步、确定性 actor，统计两只手到 B1 的最近距离）：

| ckpt（步）| 存活 | **左手**最近 B1 | **右手**最近 B1 | 距 B1 < 10 cm 步数（左/右）| `max_bar` |
|---|---|---|---|---|---|
| 0.74M | 260 | 0.280 | 0.297 | 0 / **0** | 0 |
| 3.75M | 148 | 0.171 | 0.317 | 0 / **0** | 0 |
| 6.77M | 228 | **0.015** | 0.199 | **209 / 0** | 1 |
| 9.79M | 55 | **0.003** | 0.275 | **209 / 0** | 1 |
| 12.2M | 181 | **0.000** | 0.161 | **936 / 0** | 1 |

* **左手是单调收敛的棘轮**：0.280 → 0.171 → 0.015 → 0.003 → **0.000**（最后那个 ckpt 有 936 步在 10 cm 内）。
* **右手完全没有趋势**：0.297 → 0.317 → 0.199 → 0.275 → 0.161（在 0.16–0.32 之间来回），
  而且**所有 checkpoint、全部约 5000 步里，右手一次都没有进过 10 cm**。那点 0.30→0.16 的"改善"
  只是躯干被 `x` 维拉向 B1 时把手带过去了，不是主动够。

**原因（数量级上很干净）**：右手的两个 goal 维都是"阈值型"，在远处**数值上是 0 且梯度为 0**：

| `d_R,B1` | `c_R,gk = exp(-(d/0.04)^2)` | `dc/dd` | 有方向？ |
|---|---|---|---|
| 0.05 m | 2.1e-01 | -13.1 | ✅ |
| 0.10 m | 1.9e-03 | -0.24 | 勉强 |
| 0.15 m | 7.8e-07 | -1.5e-04 | 基本无 |
| 0.20 m | 1.4e-11 | -3.5e-09 | ❌ |
| 0.35 m | 5.6e-34 | -2.5e-31 | ❌ |

`h_R,gk` 更是要"连续 30 步在 5 cm 窗内"才可能非 0。⇒ **右手在到达 ~10–15 cm 之前，goal 里没有任何一维在要求它动**；
左手之所以能进来，是它**碰巧**先荡进了那个区域（棘轮一旦被激发就单调收敛），随后策略就专门化了
（左手 `h_L,gk` 已经=1，而左手单独也永远达不到目标 ⇒ 卡在一个**没有梯度的平台**上，`dist ≈ sqrt(2) ≈ 1.41`）。

**所以对 6 h 的预测**：
* 6 h 大概会让**单手悬挂更稳**（存活更长、掉得更少，像 #13 当年那样），
* 但**不会自己长出"右手也过去"**——因为缺的不是收敛时间，而是**右手那一侧根本没有任何方向信号**，
  唯一的希望是探索恰巧把右手送进 10–15 cm 内，而这个事件在 12.2M 步里没有发生过
  （eval 里 `bar_R` 从未为正；探针 ~5000 步里 0 次）。
* 反面对照：**#13 有 `p(6)`**——它是**渐进**量（身体任何摆动都会改变它，目标分布里天然包含"更靠近 B1"的值），
  所以后手**有**可用的棘轮（#18 的右手最近 0.166–0.191 m、`max_bar` 在训练中从 0 涨到 15.8 正是这种棘轮）。

**结论与建议**：**不要把 6 h 押在 `dual6c` 上**（缺梯度，是结构问题）；要押就押在**保留 `p(6)` 的配方**上
（#18 的 6 h，或 `dual_ladmax`）——那里"后手"至少有一个渐进的、任意距离都有方向的量。### 9.64 `dual6c` vs #18 的区别，以及"`dual6c` + `p(6)`"值不值得试

**组成（实测打印，不是凭记忆）**：

| 变体 | goal 维数 | 组成 |
|---|---|---|
| `dual6c`（#19）| 6 | `[x, z, c_L,gk, c_R,gk, h_L,gk, h_R,gk]` |
| `dual_nomax`（#18）| 10 | `[x, z, p_L(3), p_R(3), h_any, h_dual]` |
| **`dual_hnext`** | **12** | **`[x, z, p_L(3), p_R(3), c_L,gk, c_R,gk, h_L,gk, h_R,gk]`** |

**两者差两处**（所以 #18 与 `dual6c` **不是干净的单变量对**）：

| | `dual6c` (#19) | #18 `dual_nomax` |
|---|---|---|
| `x, z` | ✅ | ✅ |
| **`p(6)`（抓握座世界坐标，线性、任意距离有方向）** | ❌ | ✅ |
| 瞬时接触 | `c_L,gk, c_R,gk`（**每手×目标杆**，阈值型：0.2 m 处 1.4e-11）| **无** |
| 持续性 | `h_L,gk, h_R,gk`（每手×目标杆）| `h_any, h_dual`（**杆无关**）|

真正干净的单变量对是：`dual6c` ↔ `dual6d`（同维，只换接触编码）、#18 ↔ #13（只加 `max(c)`）。

**而"`dual6c` + `p(6)`"就是 `dual_hnext`**——它**已经实现好了**（§9.52 / commit `877ef14`），从未跑过。
所以这个问题的答案是：**值得试，而且不用写新代码**；它相对 `dual6c` 只多了一样东西（`p(6)`），
正好是 §9.63 诊断出的那个缺陷的对症药。

**可检验的预测**（跑完用 `.scratch/rprobe.py` 量各 checkpoint 的"每手最近 B1 距离"，与 §9.63 的表直接对照）：

* 若 `dual_hnext` 的**右手**最近 B1 距离随训练单调下降并进入 < 0.10 m（甚至 `bar_R` 转正、`dual_runmax` 上升），
  则"后手缺的是**渐进方向量**"这一诊断成立 ⇒ 后面所有 goal 设计都必须保留 `p(6)` 或等价的线性距离项。
* 若右手仍然像 `dual6c` 那样停在 0.16–0.32 且 0 次进 10 cm，则 `p(6)` 也救不了后手，
  问题在别处（例如"单手专门化"是一个太强的吸引子，需要显式惩罚单手持有时长）。

**风险/注意**：

1. 相对 #13 它改了**两处**（`max(c)` → 每手目标杆 `c`；`h_any/h_dual` → 每手目标杆 `h`）⇒ 若失败无法归因，
   那时需要 §9.59 的 A/C 对照（`dual_ladmax` vs `dual_hnext`）。
2. 它**没有杆无关的接触项**，而 §9.62 的形态统计提示 `max(c)` 可能正是"敢松开旧杆、把支撑迁到新杆"的诱因
   —— 这是本 run 最大的风险点。
3. 阈值要用 `--goal-reach-thresh 0.8`（§9.52 实测：吊在目标杆下 0.721，错杆 2.093）。### 9.65 结果：`dual_hnext` 6 h **成功**（§9.64 的预测被证实），但终态不如 #17 干净

`runs/brach_dual_hnext_6h`（60 evals / 50,131,712 步 / 本地单卡 `--gpu 0` / `buffer_gb 1.0` / `git 51e5e28`）。
与 #17（`support_dual`，同预算）逐 eval 对照：

| eval | **dual_hnext** `len`/`dualMx`/`barR`/`fell`/`d/len` | **#17**（对照）|
|---|---|---|
| 10 | 360.9 / **60.9** / −62 / 0.31 / 1.137 | 51.9 / 15.9 / −25 / 1.00 / 1.350 |
| 15 | 392.5 / 111.3 / **+18** | 374.9 / 100.1 / +13 / 0.31 / 0.846 |
| 20 | 412.8 / 263.6 / +212 / 0.19 / 0.587 | 450.4 / 314.9 / +330 / 0.12 / 0.399 |
| **40（峰值）** | **501.0 / 398.3 / +307 / 0.00 / 0.503** | 501.0 / 421.2 / +446 / 0.00 / 0.215 |
| 60（末）| 501.0 / 313.1 / **+129** / 0.00 / 0.774 | 501.0 / **424.6** / **+432** / 0.00 / **0.198** |

* **起飞更早**（eval 10 就 `dualMx` 60.9；#17 到 eval 15 才 100.1），**eval 40 已达 `len` 501 / `fell` 0%**；
* **后 1/3 回退**：`dualMx` 398→331→313、`barR` +307→+224→+129、`single_on_steps` 56→73→162；
* 离 B1 最近距离在 **eval 44 起 = 0.000**（有手贴到杆心）并保持到结束。

**§9.64 预测检验（证实）**：终版探针（6 集）`B1最近: 左0.000 右0.000 | 距B1<10cm步数: 左1467 右413`。
对照 `dual6c`（5 个 ckpt、约 5000 步）：右手 0.16–0.32 m、**0 步 <10 cm**。
⇒ **"后手缺的是渐进方向量（`p(6)`）"这一诊断成立**，`dual6c` 的失败是结构性的（缺梯度），补上 `p(6)` 即可解决。

**终版渲染**（`runs/render/render_hnext6h`，4 集 × 220 步）：4/4 存活、`max bar=1`、
`goal dist 2.090→0.229`（12 维）、`position-only 0.605→0.041`、**每集两只手都摸到 B1**
（左 0.001–0.004 m、右 0.002–0.011 m）、`hand_switches` 19–25。

**支撑形态对比**（存活段 `(bar_L,bar_R)` 占比）：

| 策略 | 存活 | 零支撑 | 守新杆（单手在 B1）| **双持（L1R1）** | 守旧杆 |
|---|---|---|---|---|---|
| **dual_hnext 6h** | 221 | 9% | **74%** | **13%** | 2% |
| #17 `support_dual` 6h | 502 | 2% | 2% | **91%** | 0% |
| #19 `dual6c`（失败）| 115 | 45% | 17% | 0% | 20%+14% 两杆桥 |
| #18 `dual_nomax`（失败）| 128 | 30% | 7% | 0% | **55%** |

⇒ **它确实把支撑迁到了新杆**（守旧杆只有 2%，而两家失败的分别是 34%/55%）——这是 `p(6)` 补上方向后的直接效果；
但**大部分时间是"左手吊在 B1、右手在飞"（74%）**，真正的双持只占 13%，远低于 #17 的 91%。
所以 #17 的**杆无关** `max(c)` + `h_dual` 在"稳定双持"上仍然更强（§9.59 的 A vs C 问题现在不只是"能不能启动"，
还关系到**终态稳不稳**）。

**下一步的两个候选（都是单变量）**：
1. **`dual_ladmax` = `[x, z, p(6), max(c_L,c_R), h_L,gk, h_R,gk]`（11 维）**：= #13 只把 `h_any/h_dual`
   换成"每手对目标杆"。它同时保留 #17 里被证明有效的三件（`p`、杆无关 `max(c)`）与 §9.52 想修的漏洞
   （`h` 不再被起始杆白送）。**预测**：双持占比应回到 #17 的水平（>80%）；若反而掉到 13% 附近，
   说明"终态双持"靠的是 `h_dual` 的"同杆"语义，而不是 `h` 的参照物。
2. 让 `dual_hnext` 继续跑（或重跑到 100 evals）看后 1/3 的回退是否会自行恢复。

### 9.66 给"魔力"归因的两个单变量实验：`dual_hnext_max` / `dual_hnext_dual`

§9.66 前的观察：八条 run 摊开后，`max(c)` 的"魔力"看起来不是任务层面的要求，而是**抓握深度**
（渲染实测 soft grasp：#17 = 0.89/0.88；`dual_hnext` = 0.58/**0.12**；#18/#19 = 0.03–0.12）。
三条角色分工（每条缺失都会退化到某种失败）：

| 角色 | 由谁承担 | 缺了会怎样（实测）|
|---|---|---|
| ① 到达（两只手、任意距离都有方向）| `p(6)` | #20 不动；#19 后手无梯度停在 0.16–0.32 m |
| ② 抓握深度（勾住、能承力）| `max(c)`（杆无关、增益 ≈21/m、**吊着就能改善**）| #18 浅勾打滑（soft grasp 0.03–0.11）；`dual_hnext` 右手"摸到不勾"（0.12）⇒ 双持只有 13% |
| ③ 保持双手同杆（t=0 就已满足的限制项）| `h_dual` | `dual_hnext`（无 `h_dual`）双持仅 13%；#11（无 `h_dual`，只有 `h_any`）单手吸引子仍在 |

**因此把两个角色各做成"在已成功的 `dual_hnext` 上加一维"**（只加不删，各自单变量）：

| 变体 | goal（13 维）| state/obs | 检验 |
|---|---|---|---|
| **`dual_hnext_max`** | `[x, z, p(6), max(c), c_L,gk, c_R,gk, h_L,gk, h_R,gk]` | 146 / 159 | 角色 ②（抓握深度）|
| **`dual_hnext_dual`** | `[x, z, p(6), c_L,gk, c_R,gk, h_L,gk, h_R,gk, h_dual]` | 147 / 160 | 角色 ③（保持双手同杆）|

CPU 校验（t=25 稳态悬挂）：

| 变体 | 吊 B0 / goal B1 | 吊在目标杆下（浅勾）| 深勾后（c≈0.97 外推）|
|---|---|---|---|
| `dual_hnext_max` | **2.147** | **0.864** | ≈0.14 |
| `dual_hnext_dual` | **2.095** | **0.726** | ≈0.16 |
| （对照）`dual_hnext` | 2.093 | 0.721 | — |

⇒ 阈值统一用 **`--goal-reach-thresh 1.0`**（两者都满足：目标杆下 ≤0.86、错杆 ≥2.09，约 2× 余量）。
提交守卫已过（17 个变体自洽）。

**预测（跑完对照检查）**：

* `dual_hnext_max`：右手 soft grasp 从 0.12 升到 ≳0.7、双持占比从 13% 升到 **>70%**、`hand_switches` 下降
  ⇒ 角色 ② 成立（抓握深度是终态稳定的关键）。
* `dual_hnext_dual`：双持占比升到 **≳80%**，但右手 soft grasp 仍偏低（"摸到即松"）⇒ 角色 ③ 成立。
* 若**两者都只升到 ~30%**，说明真正的推手是 #17 里两者的**组合**（或 `h_any`），需要再设计。### 9.68 两杆课程：`--train-goal-bar-max`（用户提议的三种 episode）

用户提议：训练集应包含 **B0→B1、B1→B2、B0→B2** 三种 episode。原来表达不出来——
`--goal-ahead 1 --goal-ahead-max 1` 只能给前两种，`--goal-ahead-max 2` 又多出 B1→B3；
而且**随机起点 + 采样目标**在旧代码里允许采到"身后"的杆。

**改动**（`src/envs/brachiation.py` + `src/train.py`）：

* 新增 `--train-goal-bar-max`（含端点，`-1` = `n_bars-1`）；
* **当 `start_bar_max > 0` 时，采样区间收窄为 `[max(goal_bar_min, k0+1), goal_bar_max]`**
  ⇒ 指令**恒在起点之后**（消除了"朝后目标"这个隐患，见 §9.45/§9.50）；
* `--goal-ahead` 分支的 `span` 也按 `goal_bar_max` 收窄；构造环境时若
  `goal_bar_max < start_bar_max + 1` 直接报错（不可能满足"恒朝前"）；
* `args.json` 现在记录 `start_bar_max / train_goal_bar_min / train_goal_bar_max`。

**实测分布**（256 次 reset，`support_dual`）：

| 配置 | 分布 | 朝后目标 |
|---|---|---|
| **`--start-bar-max 1 --train-goal-bar-min 1 --train-goal-bar-max 2`** | **B0→B1: 57 / B0→B2: 70 / B1→B2: 129** | **0/256** |
| `--start-bar-max 0 --train-goal-bar-min 1`（旧行为）| B0→{B1..B4} 各 ~64 | 0 |
| `--start-bar-max 1 --goal-ahead 1 --goal-ahead-max 1`（旧的"只差一格"）| B0→B1 / B1→B2 | 0 |
| `--start-bar-max 1 --train-goal-bar-min 1`（无 max）| B0→{B1..B4} / B1→{B2..B4} | 0（新逻辑已强制朝前）|

守卫：`check_args` / `check_metrics` / `check_variants`（17 变体）全过。

**为什么这个课程有道理**：起点在 B1 的 episode 与起点在 B0 的**物理完全同构**（reset 只是把整机平移
`k0·spacing`，代码注释里已注明"bit-identical physics"）⇒ 策略因此看到同一套转移技能的多个"位置副本"，
而 B0→B2 那类 episode 正好提供**两格链式**的状态分布（这正是"连续过多杆"要的东西）。
注意训练期 actor 的 goal 输入取自**未来状态重标记**，指令只体现在 `goal_set[gk]`（绝对坐标 ⇒ 目标身份
因此是可区分的）。

**命令（1.5 h 侦察，20 evals；#17 配方一字不改，只加课程三个 flag）**：

```bash
.venv-warp/bin/python -u src/train.py --preset C_l2_infonce --num-envs 128 \
  --num-eval-envs 16 --batch-size 512 --min-replay-size 1000 --unroll-length 62 \
  --action-window reach --goal-variant support_dual \
  --start-bar-max 1 --train-goal-bar-min 1 --train-goal-bar-max 2 \
  --eval-goal-bar 1 --expl-hold 10 \
  --num-evals 20 --steps 12200000 --save-every 5 \
  --buffer-gb 1.0 --xla-mem-fraction 0.6 \
  --checkpoint-dir runs/ckpt_dual2bar --impl warp --scene full035 --gpu 0 \
  --wandb --exp-name brach_dual2bar 2>&1 \
  | grep --line-buffered -v -e dot_search_space -e autotuning | tee runs/brach_dual2bar.log
```

**判据**：eval 固定在 B1（与 #13/#17 可比）⇒ 看 B0→B1 的技能有没有被课程破坏（`dual_runmax`/`bar_R`/`fell`）；
再看 `max_bar` 是否 >1（链到第二格的迹象）。跑完用 `render_policy.py --goal-bar 2` 回放，
直接看它对"去 B2"这个指令的反应（渲染可以自由改指令，训练不能）。#### 9.68.1 采样改成"先抽跨几格 Δ，再抽起点"（用户要求：B0→B2 最多，两个一格近似相等）

上一版（k0 先抽、gk 再在可行范围里均匀抽）给出的是
**B1→B2 50% / B0→B1 25% / B0→B2 27%**——把"第二个一格副本"过度加权，反而把两格链式压到 27%。

改成 **Δ-first**：先在**可行距离**上均匀抽 Δ，再在该 Δ 允许的**起点**里均匀抽 k0
（可行性表在 `__init__` 里用纯静态配置算好，存成 `_delta/_klo/_khi`）。对两格课程：

* Δ=1 → k0 ∈ {0,1}（各 25%）
* Δ=2 → 只有 k0=0 可行（50%）

**实测（512 次 reset）**：

| 配置 | 分布 | 朝后目标 |
|---|---|---|
| **`start≤1, goal∈[1,2]`** | **B0→B2 50% / B0→B1 23% / B1→B2 26%** | 0 |
| `start≤1, goal∈[1,3]` | B0→B3 **31%**（最长格最多）/ B0→B2 21% / B1→B3 16% / B0→B1 15% / B1→B2 17% | 0 |
| `start=0, goal∈[1,4]`（#17 配方，无随机起点）| 23–27% 均匀（**行为不变**）| 0 |

一个附带的好性质：**最长的可行距离总是众数**（因为只有 k0=0 能承载它），所以"越远越难"的课程权重是自动的。
`start=0`（固定起点）时这条采样路径完全不触发 ⇒ 不与历史 run 冲突。

命令与 §9.68 相同（flag 未变）。守卫 `check_args` / `check_metrics` / `check_variants`(17) 全过。### 9.69 结果：两杆课程（`support_dual` + start≤1 + goal∈[1,2]，6 h）——一格优秀，两格"能链但守不住"

`runs/brach_dual2bar_6h`（60 evals / 50,131,712 步 / `--gpu 1` / git `9aa3980-dirty` / 课程三个 flag）。
eval 仍固定在 B1 ⇒ 与 #17 同预算逐 eval 可比。

**① 一格技能：没被课程破坏，eval 30 起与 #17 同级**

| eval | 课程 run `len`/`dualMx`/`barR`/`fell`/`d/len` | #17 对照 |
|---|---|---|
| 20 | 197.8 / 12.4 / +92 / 0.69 / 1.067 | 450.4 / 314.9 / +330 / 0.12 / 0.399 |
| **30** | **501.0 / 397.2 / +448 / 0.00 / 0.292** | 501.0 / 409.2 / +437 / 0.00 / 0.245 |
| 40 | 501.0 / 403.8 / +420 / 0.00 / 0.280 | 501.0 / 421.2 / +446 / 0.00 / 0.215 |
| 60 | 501.0 / 393.4 / +434 / 0.00 / 0.333 | 501.0 / 424.6 / +432 / 0.00 / **0.198** |

渲染（`--goal-bar 1`，4 集 × 300 步）：**4/4 存活**、`goal dist 1.538 → 0.038`、
两只手都在 B1（0.000–0.001 m）、`grasp 0.63/0.76` ⇒ 比 #17（`d/len` 0.198、grasp 0.89/0.88）互有高低，
但**明显优于 `dual_hnext` 终版**（74% 单手、13% 双持）。

**② 两格（渲染 `--goal-bar 2`，终版 4 集）——换手链出现了，末态崩**

| | ep0 | ep1 |
|---|---|---|
| 阶段序列 | `t0 双手B0` → `t2-18 左手守B0/右手腾空` → **`t19-30 右手抓住 B1`** → `t31 双手B1（仅 1 步）` → `t36 左手够到 **B2**` → `t37-48 左手在B2/右手离杆` → `t49 双手离杆` → **done@57** | 同构，`done@60` |
| 躯干 x | −0.017 → +0.680（B2 悬挂位 0.685）| → +0.671 |
| `goal dist` | 1.864 → 1.744（最小 **1.069**）| 1.860 → 1.740（最小 1.037）|

⇒ **`max_bar = 2`（4/4 都够到第二根杆）**，但 4/4 在 t≈58 坠落、`grasp 0.05/0.03`、
`goal dist` 从未低于 1.04（= 只有一只手在 B2，另一只手的两维各留 1.0 缺口）。

**③ 跨 checkpoint 趋势（★这条决定了"是不是只是慢"）**

| ckpt（步）| `max_bar` | 坠落 | 离 B2 最近（左 / 右）|
|---|---|---|---|
| 13.47M | **1** | 74/38/73/76 | 0.093 / 0.115 |
| 25.97M | 2 | 36/52/63/32 | **0.005** / 0.041 |
| 38.47M | 2 | 64/52/66/117 | **0.002** / 0.092 |
| 50.13M（终版）| 2 | 57/60/61/58 | **0.001/0.023/0.028/0.003** / 0.099–0.192 |

* **"够到 B2"是在训练中长出来的**（13M→26M 之间），左手最后能稳定贴住 B2（1–30 mm）；
* **右手从未接近 B2**（0.10–0.19 m），**坠落时刻在所有 checkpoint 相同（4/4、t≈60）**
  ⇒ 末态稳定这一环**卡住了，不是"训练还不够久"**。

**④ 机制读法（来自轨迹）**：它在 B1 上**只停留 1 步**就发起第二次转移（`t31 L1R1` 仅 1 步），
因为**目标是 B2 ⇒ 停在 B1 在目标距离上没有回报**。也就是说：
**缺的不是"能链"或"能守"，而是"先稳住 B1、再走 B2"这个中间目标本身没有激励**——
`support_dual` 的 `max(c)/h_any/h_dual` 都是**杆无关**的（右手待在 B1 上就已经满足它们），
所以没有任何一维要求"右手也跟到 B2"。

**⑤ 工具修正**：`render_policy.py` 的 `d_LB1/d_RB1` 列与 "closest hand to B1" 原本**硬编码 1 号杆**，
渲染 `--goal-bar 2` 时报告的其实是 B1 的距离（易误判）⇒ 已改成**相对指令杆**（`d_Lgoal/d_Rgoal` 语义）。

**下一步候选（单变量）**：
| 选项 | 做什么 | 针对的正是 ④ |
|---|---|---|
| **A** | 把 goal 换成 **goal 杆专属**的那一族（`dual_hnext`：`c_L,gk/c_R,gk/h_L,gk/h_R,gk`）**并沿用本课程** | `k_goal` 进入 state ⇒ **右手对 B2 的接触/持续被显式要求**（`support_dual` 缺的正是这一维）|
| B | **分段指令**：episode 内先给 B1，稳住后再把指令推进到 B2 | 直接给"稳住 B1"发工资（最对症，但是新机制）|
| C | 加大 B0→B2 权重 | 已经是众数（50%），边际收益存疑 |### 9.70 真实接触传感：warp 下可用的信号是 `qfrc_constraint`（实测）

用户提问：能否在手指处加触觉、用**真实接触**而不是距离判据（因为"够到 B1 后手离杆很近却不肯抓住"）。
逐条查证 + 实测（`impl="warp"`，CPU）：

| 途径 | 可用性 | 证据 |
|---|---|---|
| `data.contact`（`geom`/`dist`/`dim`/`efc_address`）| **✗** | `AttributeError: 'DataWarp' object has no attribute 'contact'` |
| `data.efc_force`（约束力向量）| **✗** | 同上（`Data` 的类型声明里有，warp 后端不填/不暴露）|
| `<touch>` 传感器（`data.sensordata`）| **✗** | MJX 的 `sensor.py` 只实现 `sensor_pos/vel/acc`（`SensorType.TOUCH` 只在枚举里）；本场景 `sensordata` 仅 12 维 |
| **`data.qfrc_constraint`（nv=49）** | **✓** | 见下 |

**实测（30 步稳态）**：

| 状态 | 手指 DOF 的 `abs(qfrc_constraint)` 合计 | 全 DOF 最大 |
|---|---|---|
| 手指闭合、双手悬垂 | **左 40.1 / 右 41.0**（N·m）| **337.5** = 34.39 kg × 9.81（**正好是体重**）|
| 手指张开（松手）| **0.00 / 0.00** | 0.10（自由落体，无约束力）|

⇒ **手指 DOF 的约束力就是"这只手在不在承力"的物理量**：悬垂 40 N·m 级、松手 0.00，干净的 40 vs 0。
（`qfrc_constraint` 也含关节限位/等式约束，但"松手"时读到 0.00 说明限位没有污染；实现时仍需在"手指闭合但悬空"状态复核一次。）

#### 9.70.1 用户提议的 `h_any_contact` / `h_dual_contact`：可实现，而且正打在病根上

**病根（本轮的另一个观察）**：现在所有"接触/持续"项**都是距离判据**——
`max(c)`（座到轴的距离）、`h_any`/`h_dual`（5 cm 窗内）——**"悬在 4.9 cm 处"就算抓住**。
所以在 goal=B1 时手可以"贴着杆不抓"；在两格情形下，右手停在 B1 就已经满足 `h_dual`/`max(c)`
⇒ **没有任何一维要求它跟到 B2 并承力**（§9.69 的机制）。力判据不可能被"悬停"满足 ⇒ 直接堵死这条捷径。

**实现草案**（软化的，避免"跳变"这个当初让计划回避接触项的问题）：

```
s_h      = tanh( Σ_{d ∈ 手指 DOF of hand h} |qfrc_constraint[d]| / F0 )     # F0 ≈ 5（悬垂≈40 → s≈1；无接触→0）
S_any    = 连续满足 max(s_L, s_R) > θ 的步数   → h_any_contact  = 1 - e^{-S_any/10}
S_dual   = 连续满足 "两手 s_h > θ 且 argmin_k d[L,k] == argmin_k d[R,k]" 的步数 → h_dual_contact
```

* 距离项（`p(6)`、`c`）继续提供**连续引导**；接触项只作**终态/持续判据** —— 正是计划 §3.2 想要的
  "几何量做引导、接触做终态"的分工，只是这次用**力**而不是 5 cm 窗；
* 两个新维度既进 **state** 也进 **goal**（C2：goal 必须是状态的一部分），`support_dual` 从 11 → 13 维；
* 阈值需重调（新维度会让"已吊在目标杆下"的 dist 变化），CPU 校验 + `check_variants` 守卫照旧。

**风险**：① `qfrc_constraint` 含限位/等式项（需复核）；② 首次触碰瞬间信号稀疏（用 10 步时间常数平滑）；
③ `F0` 的标定（按实测 40 量级取 5–10）。

### 9.71.1 接触特征的标定：`F0` 按"关键帧双手悬垂"标定，`θ` 取 0.3（+5 步 EMA 防抖）

**先说一条测量结论（它决定了标定方式）**：把手指 DOF 的载荷 `F` 按"手到**最近**杆的距离"分档
（rollout 2154 步，用两杆课程 6h 的策略，goal_bar∈{1,2}）：

| 到最近杆的距离 | n | F 中位 | F_p25 | F_p95 | F<15 占比 |
|---|---|---|---|---|---|
| ≤1 cm（深勾）| 610 | 65.6 | 53.3 | 67.6 | 7% |
| 1–2 cm | 565 | 24.7 | 17.5 | 68.7 | 21% |
| **2–3.5 cm（窗内边缘）** | 352 | **62.2** | 15.3 | 76.2 | 24% |
| **3.5–6 cm** | 247 | **62.2** | 34.1 | 71.6 | 18% |
| 6–10 cm | 151 | 41.6 | 14.1 | 72.7 | 25% |
| >10 cm（腾空）| 229 | **2.1** | 2.1 | 65.2 | 57% |

⇒ **距离根本不是接触的代理**：手离每根杆 3.5–6 cm 时载荷中位数仍 62 N·m（接触发生在掌心/手指/腕部，
而 `d` 量的是**标定座**到**杆轴**的距离），>10 cm 档里也有 65 的样本。
这既证明"必须用接触量"，也说明 **`F0/θ` 不能按距离分档标定**，要按物理量标定。

**标定方案（与仓库既有做法一致：拿关键帧当基准）**

| 参数 | 取值 | 依据 |
|---|---|---|
| **`F0`** | **40 N·m** | 关键帧**双手悬垂**时手指 DOF 的 `sum abs(qfrc_constraint)` 实测 **40.1 / 41.0**（同时全局约束力 = **337.5 N = 34.39 kg × 9.81**，即体重 ✓ 交叉验证）。`s = tanh(F/40)` ⇒ 关键帧悬垂 **0.76**、训练后更深的抓握（实测中位 50–66）**0.84–0.90**、完全无接触（实测 ~2.1）**0.05** |
| **`θ`** | **0.3**（⇔ `F ≈ 12.4 N·m`）| 距"无接触"~6 倍、距"真实抓握"~2.5 倍，两侧都有余量 |
| **平滑** | 先做 **5 步 EMA**（`s̄ ← 0.8 s̄ + 0.2 s`）再判阈 | 摆动顶点会瞬时卸载（实测 ≤1 cm 档 p25 = 53 但存在低值），M4 §8 的"判据太紧 ⇒ 真实悬垂 flicker"教训；这也和手部文档里抓握指令的"一阶低通"同一手法 |

**这两个数应当做成 CLI 参数**（`--contact-f0/--contact-theta/--contact-ema`）并写进 `args.json`
——它们是**标定尺度**而不是任务超参，换机器/换机器人可能需要重测，不该藏在代码里。

**验收判据（可测，避免再靠感觉）**：

* **正对照**：t=0..20（双手吊 B0、关键帧姿态）应有 ≥95% 的步满足 `s̄ > θ`；
* **负对照**：自由落体段（`done` 之后，或两手离所有杆 >10 cm 且身体在加速下落）应 ≤1% 步满足；
* 两者都进 `cov_contact_*` 指标，**每次 run 都能看到标定是否健康**。

**顺带一个重要旁证**：既然"座离轴 5 cm 仍可能实打实挂着"，那么现在所有基于 `dmin`/5 cm 窗的判据
（`h_any`、`h_dual`、`max(c)`）**比我们原先以为的更弱**——它们既可能把"真挂着"判成"没抓"，
也可能把"贴着但不承力"判成"抓住了"。接触量两头都修。

### 9.72 实现并验收：真实接触项 `f_{h,gk}`（`dual_hcontact` / `support_dual_contact`）

按 §9.70.1 的草案 + §9.71.1 的标定实现完毕。**两处对 §9.71.1 的修正**写在最前面，因为它们影响读数方式。

#### 9.72.1 修正一：40.1/41.0 是**静态关键帧**的值，控制回路里的悬垂是 **48–68 N·m**

§9.70 的 40.1/41.0 是在关键帧上取一帧 `qfrc_constraint` 得到的。让机器人**真的通过 `env.step`
（位置伺服 + 10 帧物理子步）吊住**之后，手指 DOF 的 EMA 载荷是：

| 状态（`src/check_contact_sense.py` 实测，full035）| 手指载荷 (N·m) |
|---|---|
| 双手闭合悬垂 B0（关键帧姿态，正对照）| min **48.1** / p50 65–67 / max 68 |
| 单手悬垂（左手张开、右手承力）支撑手 | min **37.7**（冷启动后）/ p50 59 / max 68 |
| 单手悬垂时的**悬停手**（座在 B0 窗口内、手指张开）| max **9.5**，p50 **3.1** |
| 松手后（跳过释放瞬态）| max **0.9** |
| 闭合但**抓在错误的杆**上（goal=B1 而机器人在 B0）| min **52.8**（载荷照旧，只是杆不对）|

⇒ `F0 = 40` 作为**标定基准**仍然合理（它复现了关键帧、且 337.5 N = 体重交叉验证），
但"真实抓握"的实际下沿是 **37.7**（单手摆动）而不是 40。于是阈值的可行区间是：

* **下界**：手指张开的"轻擦"峰值 **9.5** ⇒ `θ > 9.5/40 = 0.24`，否则悬停会被判成抓住；
* **上界**：单手摆动悬垂的最低点 **37.7** ⇒ `θ < 0.47`，否则真实承力会 flicker（θ=0.4 已开始丢步，
  θ=0.5/0.6 连双手悬垂的 `obs` 都会因为冷启动+摆动掉到 0.85–0.91）。

**取 `θ = 0.3`（阈值 12.0 N·m）**：高于轻擦 1.26×，低于单手摆动 3.1×，低于双手悬垂 4.0×。

**θ 扫描（18 条对照全过 = 可行）**，实测表（`--contact-theta`，其余不变，20 步/臂）：

| `θ` | 阈值 | 18 条对照 | 失败的那条说明什么 |
|---|---|---|---|
| 0.20 | 8.0 | ❌ 2 条 | 轻擦峰值 **9.5** 被算作承力 ⇒ 悬停被判成抓住 |
| **0.25** | 10.0 | ✅ 全过 | 刚好越过轻擦 |
| **0.30** | **12.0** | ✅ 全过 | **采用值** |
| **0.40** | 16.0 | ✅ 全过 | |
| **0.45** | 18.0 | ✅ 全过 | 2× 裕度的上沿 |
| 0.50 | 20.0 | ❌ 1 条 | 单手摆动悬垂最低 **37.7** 不再有 2× 裕度 |

⇒ 可行带 **θ ∈ [0.25, 0.45]**（阈值 10–18 N·m），0.3 在其中且偏"绝不把悬停当抓住"一侧；
这是**实测出来的带**，不是刀尖上的取值。

#### 9.72.2 修正二：特征不能是"载荷超过阈值"，必须是 **(在目标杆窗内) AND (载荷超过阈值)**

载荷本身**说不出是哪根杆**（P2 对照：抓在 B0、goal=B1，载荷 52.8 照旧）。所以：

```
F_h   = EMA_{5}( Σ_{d ∈ 手 h 的 7 个手指 DOF} |qfrc_constraint[d]| )        # N·m
s_h   = 连续满足 ( d[h, gk] < 0.05 m ) AND ( F_h > F0·θ ) 的步数
f_{h,gk} = 1 - exp(-s_h / 10)                                             # 与 hnext 同一映射
```

* 距离门 = "这根杆"，力判据 = "真的在承力"，两者**缺一不可**；
* 这也顺带回答了"要不要 `h_any_contact`/`h_dual_contact`"：`max(f_L,f_R)` 与 `min(f_L,f_R)`
  就是**杆无关**的 any/dual 版本，而且比它们多带了杆身份 ⇒ 两维已覆盖三种读法，不需要四维。

#### 9.72.3 两个变体（都是一次只改一个变量）

| 变体 | goal | 与谁对照 |
|---|---|---|
| **`dual_hcontact`** | `[x, z, p(6), c_L,gk, c_R,gk, f_L,gk, f_R,gk]`（12 维）| **`dual_hnext`** 的单变量替换：state 148 / goal 12 / obs 160 **完全相同**，只把两个 `h_·,gk`（距离判据）换成 `f_·,gk`（力判据）|
| **`support_dual_contact`** | `[x, z, p(6), max(c), h_any, h_dual, f_L,gk, f_R,gk]`（13 维）| §9.69 两杆课程配方（`support_dual`，11 维）+ 同样两维 |

`h_any`/`h_dual` 在 `dual_hcontact` 里保留为**可观测量**（不进 goal），正是为了让它的 state 与
`dual_hnext` 逐维对齐——否证实验必须只改一个变量。

#### 9.72.4 验收：正/负/悬停/错杆 四对照（`src/check_contact_sense.py`，全部走 `env.step`）

四条臂都由**关键帧姿态 + 动作向量**驱动（`a[:12]=0` 精确复现关键帧手臂 ctrl，
`a[12+i]=2c-1` 复现抓握协同，`c` = 手指闭合量），因此不经渲染、不绕代码：

| 臂 | 设置 | `contact_L/R` | `near_L/R` | `hover_L/R` | 载荷 |
|---|---|---|---|---|---|
| **P** 正对照 | 双手闭合，goal=B0 | **1.00 / 1.00** | 1.00 / 1.00 | 0.00 / 0.00 | 48–68 |
| **H** 悬停对照 | 左手张开、右手闭合，goal=B0 | **0.00 / 1.00** | **0.37 / 1.00** | **0.37 / 0.00** | 左 ≤9.5，右 ≥37.7 |
| **N** 负对照 | 双手张开、跳过 15 步释放瞬态 | **0.00 / 0.00** | 0.00 / 0.00 | 0.00 / 0.00 | ≤0.9 |
| **P2** 错杆对照 | 双手闭合，goal=B1（人在 B0）| **0.00 / 0.00** | 0.00 / 0.00 | 0.00 / 0.00 | 48–68（照旧承力）|

**H 臂是本轮的关键证据**：左手在 B0 的 5 cm 窗内停驻 **11/30 步**，此时
`hnext`（距离判据）的 `1-exp(-s/10)` **峰值到 0.667**，而 `f`（力判据）**峰值仍是 0.000**。
把"手放进窗口"从"抓住"里剥出来，正是 §9.70.1 说的病根：
**goal=B1 时左手贴着杆不抓、以及 `dual_hnext` 后期右手 0.000 m 却 grasp 0.12**，都是这一件事。
（注意：若策略**持续**把手停在窗内——`dual_hnext` 的右手就是 **413 步**——`hnext` 会一路涨到 ~1.0，
所以这不是"峰值 0.667 无所谓"，而是"轨迹越长越骗得越狠"。）

18 条断言全过（含"每一维都进 `obs`"的接线检查：`obs[contact]` 在 P 中 0.97、在 H/N/P2 中按预期为 0）。
复现命令见 `docs/复现环境.md` §4。

#### 9.72.5 顺带修掉的**两个**真 bug（第二个是写这节时被新守卫抓出来的）

**(a) `render_policy.py` 的 `uses_hpair` 被覆盖 ⇒ 四份渲染的 `dist` 报错**

`src/render_policy.py` 里逐变体手拼 `_achieved_goal` 参数的分支中，
`uses_hpair = variant_cfg in (...)` 的下一行被一句遗留的 `uses_hpair = variant_cfg == "dual_hnext"`
**覆盖**了。后果：`dual6c`/`dual6d`/`dual_hnext_max`/`dual_hnext_dual` 渲染时 `hnext` 被当成 0，
`dist` 与 `trace.csv` 的 `dist` 列被高估（每只手最多 +1，两维最多 +√2 ≈ 1.41）。

* **受影响**：`render_dual6c_final`、`render_dual6d_final`、`render_hnext_max`、`render_hnext_dual`
  的 `dist` / `dist_pos` 数值；它们的行为列（`bar_L/bar_R/d_LB1/d_RB1`）与结论**不受影响**；
  `render_hnext6h`（真正的 `dual_hnext`）本来就走对了分支，不受影响。
* **修法**：新增 `env._achieved_goal_info(data, info)`——**唯一**入口，从 `info` 里取全部历史量，
  `step` 与渲染都调它；整个 `uses_*` 分支群被删掉（`vmap` 直接接受 `(pipeline_state, info)` 字典 pytree），
  这类"漏传一个历史量"的 bug 从结构上消失。

**(b) 新守卫 `check_variants.py --deep` 立刻抓到同一个 bug 的第二个实例（我自己写的）**

新入口第一版对 `hold` 用的是 `info["hold_streak"]`，但 `step` 对 **`advance`** 用的是
`info["next_streak"]`（"下一根杆的持续接触"，见 §9.35）。于是 `advance` 的渲染 `dist` 会算错
—— **正是 (a) 的同一种错，只是换了个变体**。深守卫（3 步真实 `step` 后比对
`‖_achieved_goal_info(...) − goal‖` 与 `metrics["dist"]`）当场报出 `1.173148 vs 0.961814`。

终态修法是**记录已解析的值**，而不是读的时候再推导一次：`step` 把它本步真正用的 `hold`
写进 `info["hold_feature"]`，`_achieved_goal_info` 直接读它。19/19 变体现在 `dist` 逐位一致
（`check_variants.py --deep`，实测输出见 `docs/复现环境.md` §2.4）。

> **教训（值得写进流程）**：goal 的"历史量"参数一旦有多于一个来源，就必须**记录**而不是**重算**。
> 静态形状守卫（原 `check_variants`）抓不到这类错误——它只看维度；只有"跑几步再和 `step` 自己的
> `dist` 对账"才抓得到。所以 `--deep` 现在上 GPU 前建议一起跑。

#### 9.72.6 下一步（待跑，1.5 h）

两个候选臂，预算与 #13/#19/#20 同为 20 evals / 1.22e7 步，便于逐 eval 叠曲线：

```bash
# A) 单变量否证：dual_hnext 的后期回退（右手贴杆不闭合）能不能被力判据修掉
.venv-warp/bin/python -u src/train.py --preset C_l2_infonce --goal-variant dual_hcontact \
  --train-goal-bar 1 --eval-goal-bar 1 --scene full035 --num-envs 128 --exp-name brach_hcontact_b1

# B) 两杆课程配方 + 接触维（用户要的 B0->B2 主线）
.venv-warp/bin/python -u src/train.py --preset C_l2_infonce --goal-variant support_dual_contact \
  --train-goal-bar -1 --train-goal-bar-min 1 --train-goal-bar-max 2 --start-bar-max 1 \
  --eval-goal-bar 2 --scene full035 --num-envs 128 --exp-name brach_contact_2bar \
  --buffer-gb 1.0 --xla-mem-fraction 0.6
```

判读仍按 §9.61/9.62 的行为指标：`bar_L/bar_R`、`max_bar`、`cov_dual_runmax_improve`、`fell`、
`len`，外加**新的** `cov_hover_*`（近而无力 = 悬停步数）与 `cov_load_*`（标定是否健康）。
`success`/`dist` 与既有变体不可横向比（goal 维数不同）。

**读数注意（和 `episode_dist` 同一条规矩）**：`eval/episode_cov_load_*` 是**每 episode 的求和**，
要除以 `eval/avg_episode_length` 才是平均 N·m；`cov_contact_*_on_steps` 同样求和，除以长度是"承力步占比"。
一个方便的恒等式：`cov_load_L / cov_contact_L_on_steps` = **该手承力期间的平均载荷**（两者同为求和）。
（未训练策略的 smoke 实测：`cov_load_L ≈ 233`、`avg_episode_length ≈ 17` ⇒ 均值 ~14 N·m，
因为随机动作第 1 步就把抓握协同压到 `c≈0.5`、机器人几帧内脱手——这是**预期**的，不是特征坏了。）

#### 9.72.7 端到端验证（CPU，全部通过）

| 检查 | 命令 | 结果 |
|---|---|---|
| 形状/自洽 | `check_args` / `check_metrics` / `check_variants`（19 变体）| 全 OK |
| **goal 读数一致性** | `check_variants --deep`（19 变体各跑 3 步真实 `step`）| 19/19 `dist` 逐位一致（抓到并修掉 §9.72.5(b)）|
| **接触特征** | `check_contact_sense`（4 臂 × 18 断言，两个变体 × 20/30 步）| 4/4 组合全过 |
| θ 可行性带 | `--contact-theta` 扫 6 个值 | `[0.25, 0.45]` 全过，带外如预期失败 |
| 训练接线 | `train.py --smoke --goal-variant {support_dual_contact,dual_hcontact}` | 两变体 `goal=13/12、state=146/148、obs=159/160`，eval CSV 里 10 个 `cov_contact/near/hover/load` 列都在，**指标全有限** |
| 上游 patch | `tools/make_patch.sh --check` | byte-identical（新增 evaluator 白名单 10 项）|

**smoke 的 NaN 不是这次改动引入的**：同一 smoke 用既有变体 `support_dual` 也走到同一个
`[abort] training diverged to NaN`（§9.34.5 已记录的 smoke 配置问题：4 envs / batch 100 / replay 50）。
判据是**环境指标是否有限**——三个 run 的 `episode_dist`/`cov_*` 全部有限，NaN 只出现在 `critic_loss`。

### 9.73 A 臂结果（`dual_hcontact` 1.5 h / 20 evals）：**阴性，但被一个我自己漏掉的 flag 污染**

`runs/brach_hcontact_b1/`（`git_commit 3b77ee7`，12,197,632 步，`--train-goal-bar 1 --eval-goal-bar 1`，
thresh 0.8，`buffer_gb 1.0`，`xla 0.6`，`--gpu 0`）。

> ⚠️ **第一个要说的事实**：`args.json` 里 `expl_hold = 1`。我给 A 的命令漏了 `--expl-hold 10`，
> 而**§9.28.1 之后所有会动的 run 都是 10**（#11/#13/#15/#16/#17/#18/#19/#20/#21/两杆 6h 全是 10，
> 只有 #7 之前的 #3/#5 没有这个 flag）。这是我的命令错误，下面 §9.73.3 给出修正后的命令。

#### 9.73.1 同预算对照（各 run 自己的 eval 19；#21 是 6 h run 的同预算点 eval 14 ≈ 11.7 M 步）

| run | variant | len | fell | **换手/步** | `max_bar` | `dual_runmax` | `bar_R` | `advance_max` | d/len | succ |
|---|---|---|---|---|---|---|---|---|---|---|
| **`brach_hcontact_b1`** | **`dual_hcontact`** | **63.6** | **1.00** | **0.151** | **1.8** | **2.8** | **−39.2** | **0.0** | **2.283** | **0** |
| `brach_hnext_max_b1` | `dual_hnext_max` | 54.1 | 1.00 | 0.154 | 4.9 | 1.2 | −46.5 | 0.0 | 2.431 | 0 |
| `brach_dual6c_b1` | `dual6c` | 145.5 | 1.00 | 0.107 | 130.3 | 2.9 | −112.4 | 96.1 | 1.741 | 0 |
| `brach_hnext_dual_b1` | `dual_hnext_dual` | 208.8 | 0.69 | 0.090 | 178.5 | 19.2 | +85.4 | 139.9 | 1.669 | 36.5 |
| `brach_dual_hnext_6h` @14 | `dual_hnext` | 392.5 | 0.25 | — | 372.7 | 111.3 | +18.4 | — | — | 190.7 |
| `brach_dual_hnext_6h` @19 | `dual_hnext` | **412.8** | **0.19** | **0.038** | **391.2** | **263.6** | **+212.5** | **334.2** | **0.587** | **295.3** |
| `brach_dual_b1_6h` @19 | `support_dual` | 450.4 | 0.12 | 0.043 | 425.8 | 314.9 | +330.1 | 395.1 | 0.399 | 311.5 |

（`max_bar`/`bar_R`/`dual_runmax` 都是**逐步求和的运行量**：`max_bar ≈ 1.8` 就是"整个 episode 只有约 2 步
有手进入 B1 的 5 cm 窗"，而 391 就是"约 390 步"。）

⇒ A 落在 **`dual_hnext_max` 那一档**（fell 100%、从不离开 B0、`advance_max = 0`），
比同族的 `dual_hnext` 差一个数量级。**这是一个阴性结果。**

#### 9.73.2 渲染（`--goal-bar 1`，4 episodes × 261 步，`runs/render/render_hcontact_b1/`）——失效发生在"到达"之前

新加的 `live` 掩码 + `f/near/load/hover` 列（见 §9.73.4）给出逐 episode 的存活期读数：

| ep | live 步 | fall | 存活期内 min `d_LB1` | 步数 `d_LB1<5cm` | min `d_RB1` | `f_L>0.5` | `f_R>0.5` | `load_L` 均值 | `load_R` 均值 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 67 | 67 | 0.042 | 2 | 0.173 | **0** | **0** | 4.8 | **56.7** |
| 1 | 67 | 67 | 0.046 | 1 | 0.166 | **0** | **0** | 6.9 | **57.3** |
| 2 | 64 | 64 | 0.071 | 0 | 0.166 | **0** | **0** | 6.0 | **58.6** |
| 3 | 67 | 67 | 0.034 | 2 | 0.174 | **0** | **0** | 7.2 | **58.0** |

**读法**：右手**实打实地抓着 B0**（载荷 57 N·m，全程），左手能伸到离 B1 的**座 3.4–7.1 cm**
（`dist_pos` 从 0.605 降到 0.251，位置子目标确实在改善），但**只有 0–2 步进过 5 cm 窗**，
`f_{L,B1} > 0.5` 在**四个 episode 里一次都没有出现**，然后 t≈67 全部坠落。

⇒ **A 的失效不是"到了 B1 却不抓"，而是"根本没到 B1"**——和 §9.28 的 run #6（`expl-hold 1`，"乱抓+掉"）
是同一个画面的量化版本。它的 `max_bar` 只有 ~2 步、`advance_max = 0`。

**同时这张表说明特征本身在按设计工作**：右手承力 57 N·m 但 `near_R = 0`（它抓的是 B0，不是指令杆）
⇒ `f_R = 0`，**没有给错杆送分**；左手在 B1 窗口附近但载荷只有 5–7 N·m（< 阈值 12）⇒ `hover_L` 记上、
`f_L = 0`，**没有把悬停当抓住**。也就是说 §9.72 的验收性质在真实策略轨迹上复现了；
失败的是**策略没学出来**，不是特征判错。

#### 9.73.3 怎么归因：一个确定的污染 + 一个待检验的机制假设

**(a) 确定的污染：`expl_hold`。** 证据不只是"别的 run 都用 10"：

| run | `expl_hold` | 换手/步 | len | 现象（§9.28） |
|---|---|---|---|---|
| run #6 `brach_reach_b1_short` | （flag 尚不存在 = 1）| **0.175** | 60 | "乱抓 + 掉"，抓握被打散 |
| run #7 `brach_reach_b1_a2p` | 10 | — | 67 | **够杆解决**（C(0.1)=1.00、摆幅 0.067→0.33）|
| **A `brach_hcontact_b1`** | **1** | **0.151** | **63.6** | 同 #6 档 |
| 会动的那些（`dual_hnext`/`support_dual`）| 10 | **0.038–0.043** | 413–450 | 保持双持、推进多杆 |

A 的换手率 0.151 与 #6 的 0.175 同档、比会动的 run 高 **3.5–4 倍**。所以**A 的阴性结果不能用来否证接触维**。

**(b) 待检验的机制假设（不是结论）**：即使补上 `expl_hold`，`f` 也许天生比 `hnext` 更难学：

* `hnext` 是**稠密**的——手靠近窗口，`S_h` 就开始涨，1−e^{−S/10} 立即给出中间值，重标记 goal 里有梯度；
* `f` 被**载荷门**二值化——在"还没有任何持续抓握"之前，`f` 在 achieved 里恒 0，
  而重标记 goal 取自 buffer 里的未来状态，**那些状态里 `f` 也几乎恒 0** ⇒ 该维在对比任务里
  近似**恒等/退化**（§9.55 的 `categorical_accuracy 0.011 vs 0.354` 就是这种信号），
  而指令 goal（`f=1`）在学会之前不可达。它会自我锁死：**要先抓住 25 步才会有 `f`，而不是靠 `f` 学会抓**。

这两条给出一个**可判别的实验**：把 A 原样重跑（只补 `--expl-hold 10`）。
* 若它到达 B1 并出现 `f>0.5` 的持续段 ⇒ (a) 是主因，接触维可用；
* 若它仍 `advance_max=0` ⇒ (b) 得到支持，下一步应给 `f` 加一个**稠密前驱**（例如
  `min(c_{h,gk}, load 的连续量)` 或把 `f` 的 streak 门放宽成"载荷的连续值"），而不是直接否决接触量。

#### 9.73.4 顺带修掉的两个 render 侧问题（都已改进，见 `src/render_policy.py`）

1. **"整段平均"把坠落尾巴算成行为**——又一次。渲染循环不做 auto-reset，坠落之后还有几百步自由落体，
   于是 `grasping (soft)` 这种全段均值被拖到 ~0（A 的第一版输出就是 `L 0.01 R 0.03`，
   而存活期内是 `L 0.06 R 0.24`、最好一步 `L 1.00 R 1.00`）。现在所有读数都**只在 `live` 掩码上取**
   （`live` 逐步记录"本步开始时还活着"），并打印 `live steps`。
   > 这个坑我踩过两次（§9.61/§9.62 的"鱼跃"统计），所以这次直接在代码里写死，不靠自觉。
2. **`strict live` 掩码本身第一版写错了**：`live = ~ever_done` 里的 `ever_done` 只是**当前累加器**
   （形状 `(E,)`），不是逐步历史 ⇒ 掩码退化成"最终全部坠落"⇒ `live steps = 0 of 261`，
   并在下一行布尔索引时抛异常。改成在 `snapshot` 里逐步记录（`rec["live"][i] = ~ever_done`，即**本步
   `done` 之前**的状态）。
3. 新列：`f_L,f_R,near_L,near_R,load_L,load_R,hover_L,hover_R`（只在接触族变体的 trace.csv 里出现），
   以及对应的 summary 行（REAL grip / HOVER / finger load EMA）。

#### 9.73.5 修正后的 A 臂命令（1.5 h，单变量：只补 `expl_hold`）

```bash
cd ~/monkeyBars_CRL && unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES
.venv-warp/bin/python -u src/train.py --preset C_l2_infonce \
  --goal-variant dual_hcontact --goal-reach-thresh 0.8 \
  --train-goal-bar 1 --eval-goal-bar 1 --scene full035 \
  --num-envs 128 --num-eval-envs 16 --episode-length 501 --batch-size 512 \
  --steps 12200000 --num-evals 20 --expl-hold 10 \
  --buffer-gb 1.0 --xla-mem-fraction 0.6 --gpu 0 --save-every 5 \
  --exp-name brach_hcontact_b1_eh10 --checkpoint-dir runs/ckpt_hcontact_b1_eh10 \
  --wandb --wandb-group .
```

判读只看三件事：`advance_max`/`max_bar` 是否离开 0、`cov_contact_*_on_steps` 是否出现持续段、
渲染里 `f_{L,B1}>0.5` 的步数是否非零。**在它跑出来之前，不要对接触维下结论。**

### 9.74 B 臂结果（`support_dual_contact` 两杆课程，6 h / 60 evals）：**B1 上成功，B0→B2 仍未解决**

`runs/brach_contact_2bar_6h/`（dual4090，19,838 s ≈ 5.5 h，`git 3b77ee7`，50,131,712 步，
`--train-goal-bar -1 --train-goal-bar-min 1 --train-goal-bar-max 2 --start-bar-max 1 --eval-goal-bar 1`，
thresh 0.35，**`--expl-hold 10`** ✓，`buffer_gb 1.0`，`xla 0.6`）。

> **读数提醒（我这轮又踩了一次）**：`eval/episode_cov_*_on_steps` 是**每 episode 求和**，
> 不是比例。`ctL = 397` 意思是"约 397 步满足"，除以 `avg_episode_length` 才是占比。
> 而且 `bar_L/bar_R` 是**带符号和**（`bar=1` 加 1、`bar=0` 加 0、不在任何窗内加 **−1**），
> 所以 `bar_L = 337.8` 与 `near_L = 397` 并不矛盾（差出来的 ≈59 是"离开 B1 窗"的步数）。

#### 9.74.1 曲线：**很晚才起飞（eval ≈47 / 约 39 M 步）**，但起飞后质量是干净的

| eval | len | fell | 换手/步 | `max_bar` | `bar_L` | `bar_R` | `advance_max` | `ctL` | `nrL` | `ctR` | `nrR` | `hvL` | `hvR` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20（= A 臂的预算）| 246.5 | 1.00 | 0.146 | **0.0** | −78.6 | −72.1 | **0.0** | 0 | 0 | 0 | 0 | 0 | 0 |
| 30 | 54.8 | 1.00 | 0.135 | 0.0 | −41.3 | −22.5 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 47 | 52.4 | 1.00 | — | 11.1 | −27.6 | −37.6 | 0.0 | 4.1 | 4.1 | 1.1 | 1.1 | 0 | 0 |
| 50（起飞）| 143.9 | 0.94 | — | 114.6 | +16.2 | −103.2 | 88.9 | 70.8 | 70.8 | 4.8 | 16.7 | 0 | 11.9 |
| 55 | 417.6 | 0.19 | — | 383.3 | +305.6 | −208.8 | 325.8 | 354.3 | 354.3 | 28.8 | 97.1 | 0 | 68.3 |
| 58 | 472.1 | 0.06 | — | 521.4 | +378.5 | −94.1 | 397.6 | 421.2 | 421.2 | 176.3 | 177.6 | 0 | 1.3 |
| **59** | **472.2** | **0.06** | 0.100 | **534.1** | **+337.8** | −18.8 | **357.8** | **397.4** | **397.4** | **217.4** | **217.4** | **0** | **0** |

两条关键读数：

1. **`ctL == nrL` 与 `ctR == nrR` 到小数逐位相等**（397.4/397.4、217.4/217.4）——即
   **"手在指令杆的 5 cm 窗内"的每一步都同时是"承力"的**，`hvL = hvR = 0`
   （整段后期**没有任何一步**处于"近而无力"）。这正是 §9.72 想要的性质，而且是在**自学的策略上**成立的。
2. `max_bar = 534`（`len` 只有 472）⇒ 运行最大值在最后约 60 步达到了 **2**：左手的座**碰到了 B2**，
   但 `k_ref` 只到 1（`advance_max = 357.8` = 持续抓握 B1 的步数），即 B2 那一段没有撑到 25 步。

**代价**：起飞比 `support_dual` 基线的两杆 6 h 晚得多（基线 eval 20 就已 `len 197.8 / max_bar 157.5`），
本臂要到 **eval 47–50（39–42 M 步）** 才动。这与 §9.73.3(b) 的猜想一致：力判据让该维**稀疏**，
需要更多数据才能自举；但一旦自举，抓握质量更干净。

#### 9.74.2 渲染 `--goal-bar 1`：**4/4 存活 501 步，真实承力抓握 82%，悬停 0%**

`runs/render/contact_2bar_g1/`（4 episodes × 502 步）：

| 读数 | 本臂 | 基线 `dual2bar_6h`（同探针）|
|---|---|---|
| 存活 | **4/4 全 502 步**（`fell` 0.06 @eval）| 4/4 存活，`fell` 0 |
| `max_bar` | **2** | 1（≈B1）|
| goal dist | 2.090 → **0.067**（13 维）| 1.538 → 0.038 |
| **`f_{·,B1}>0.5` 的步占比** | **L 82%，R 30%** | 无此读数（当时没有接触特征）|
| **HOVER（在 B1 窗内但不承力）** | **L 0%，R 0%** | — |
| 在 B1 窗内的步占比 | L 86%，R 47% | — |
| 手指载荷 EMA | L **63.5**（max 73.7）/ R 24.5（max 66.3）N·m | — |
| soft grasp（存活期）| L 0.73 / R 0.28 | L 0.63 / R 0.76 |

⇒ **指令杆上"够到就松手/贴杆不抓"这个病，被力判据治好了**：左手 82% 的存活期是真的抓着 B1 承力
（63.5 N·m），且**没有一步是悬停**。与基线相比：右手在 B1 上的时间少一些（0.28 vs 0.76），
但本臂多出的是"承力"这个语义，而不只是"座在 5 cm 内"。

#### 9.74.3 渲染 `--goal-bar 2`：**仍然失败**——链能到 B2，但守不住（比基线多活一倍）

`runs/render/contact_2bar_g2/`：

| 指标 | 本臂（`support_dual_contact`）| 基线（`support_dual`）|
|---|---|---|
| 存活 | 4/4 **t = 130 / 116 / 118 / 119** | 4/4 **t ≈ 57 / 60 / 61 / 58** |
| `max_bar` | 2（4/4）| 2（4/4）|
| 哪只手到 B2 | **左手**（min 0.010 / 0.067 / 0.058 / 0.075）| 右手（min 0.009–0.015，4/4 都到）|
| 到 B2 的步数（`d<5cm`）| 8 / 0 / 0 / 0 | **1 / 1 / 1 / 0**（左手）、**18 / 19 / 19 / 19**（右手）|
| **B2 上的真实承力抓握** | ep0 **有，但只有 ~2 步**（`load_L 31.5` N·m、`f_L 0.19`、`c_L 0.50`）；ep1–3 完全没有 | 无（`c_L@<5cm ≈ 0.4`，即"座在 3 cm 内但没咬住"）|
| 载荷去哪了 | L 34.7 / R 57.4 N·m —— 都花在**身后的杆（B0/B1）**上 | 同 |
| goal dist | 2.338 → best 1.959（位置部分 1.212 → 0.439）| 从未低于 1.04 |

**结论（这是本轮的答案）**：

1. 用户提议的 `h_any_contact` / `h_dual_contact`（实现为**杆相关**的 `f_{h,gk}`）
   **确实修掉了它针对的那个病**：goal=B1 时手不再"贴着杆不抓"——`ct ≡ nr`、`hv = 0`、
   真实承力占比 82%、4/4 存活 10 s、末端距离 0.067。
2. 但它**没有解决 B0→B2**。失效点从"在指令杆上悬停"变成了"**到了第二根杆却咬不住**"：
   ep0 左手的座到 B2 只有 **1.0 cm** 并且真的承了 31.5 N·m，但只维持了 ~2 步就坠落；
   其余三个 episode 连 5 cm 都进不去。**"到达"已经在了，缺的是"到达之后把这一抓撑住"**。
   相比基线它多活了**一倍**（116–130 步 vs 57–61），所以不是退步，但也不是解法。
3. 因此 §9.69 的下一步三选项仍然有效，而本臂的结果把候选缩小了：
   **A（目标杆专属的"撑住"压力）/ B（分阶段指令 B1→B2）** 比 C（加大 B0→B2 权重）更对——
   因为权重已经是 50%，而病不在"没练"，在"练了也守不住"。
   具体到这一维的修法：给 `f` 增加**"到达后 k_ref 才前进"**式的门（或对 `gk` 那一维要求
   `f` 与 `k_ref` 同时满足），也就是把 §9.35 `advance` 的"到达并留下"耦合到接触上。

#### 9.74.4 产物

* `runs/render/contact_2bar_g1/contact_2bar_g1.gif`（4 路并排，5.2 s/段）+ `trace.csv`（含 `f/near/load/hover`）
* `runs/render/contact_2bar_g2/contact_2bar_g2.gif`（10 s/段）+ `trace.csv`
* `runs/brach_contact_2bar_6h/progress.csv`（60 evals）+ `runs/ckpt_contact_2bar_6h/`（12 个 checkpoint）

### 9.75 「Bar-specific 的 contact 会跳过 B1」——实测确认，以及 bar-agnostic 版的实现

用户观察：**只针对 B2 的 contact 项让机器人直接扑 B2、跳过 B1**。用已有的两份 trace 直接量：

| `--goal-bar 2`（存活期）| 先碰 B2 前在 B1 上的步数 | 腾空占比 | 最长连续腾空 |
|---|---|---|---|
| **新**（`support_dual_contact`，bar-specific）| **0 / 0 / 0 / 0**（4/4 都是 0）| **56%** | 27–32 步 |
| 基线（`support_dual`，bar-agnostic，50 M）| **17 / 18 / 19 / 18** | 12% | 2–6 步 |

新臂的 run 序列是 `00x10 0-1x7 -1-1x2 0-1x16 -1-1x3 … 2-1x4`：右手吊 B0、左手乱摆，
然后**两手同时松开 27–32 步**做弹道式前扑，只在最后碰到 B2。**"跳步"是真的**，
原因是 bar-specific 项把目标写成"在 B2 上承力"，B1 一分不拿。
（对照：同一个策略 `--goal-bar 1` 时腾空只有 10%，并且把 B1 抓了 427/502 步——
所以它不是不会链，是**目标没让它链**。）

#### 9.75.1 「没有 contact 项只是训练时长不够？」——一半对

把基线 `ckpt_dual2bar_6h` 的 **B2 探针**按训练步数排开（同一渲染脚本）：

| 步数 | max_bar | 存活 | 到 B2 最近距离 | 腾空占比 | 先碰 B2 前在 B1 的步数 |
|---|---|---|---|---|---|
| 13.5 M | 1（**从未到 B2**）| t 38–76 | 0.067–0.545 | 36% | 0 |
| 26.0 M | 2 | t 32–63 | **0.005** | 8% | 18 / 19 |
| 38.5 M | 2 | t 52–117 | **0.002** | 22% | 17 / 19 |
| 50.1 M | 2 | t 57–61 | **0.001** | 12% | 17–19 |

⇒ **"到达"确实靠训练时长解决**（0/4 → 4/4 个 episode 摸到 B2；最近距离 0.067 → 0.001 m；
16→50 M 步之间还在改善），但**"守住"一点没改善**：`fell` 始终 100%、存活始终 57–61 步。
所以不是"再跑久一点就有"。

#### 9.75.2 缺的到底是什么：把载荷量出来

给渲染脚本加了"**所有变体都输出 `f/near/load/hover`**"之后，基线 @50 M、`--goal-bar 2`、存活期、
**左手**在 B2 上的读数（`near` = 座在 5 cm 窗内；`f>0.5` = 真承力）：

| ep | 在窗内 | **真承力** | **纯悬停** | 在窗内的载荷 | 在窗内的 `c` |
|---|---|---|---|---|---|
| 0 | 25% | **0%** | 8 步 | 13.0 N·m | 0.54 |
| 1 | 33% | 8% | 9 步 | 16.3 N·m | 0.71 |
| 2 | 25% | 15% | 0 | 25.8 N·m | 0.47 |
| 3 | 26% | 9% | 4 步 | 22.4 N·m | 0.82 |

⇒ 手有 **1/4–1/3 的时间在 B2 的窗内，但其中只有 0–15% 真的承力**，而承的那点载荷是
**13–26 N·m 的轻蹭**（真实悬垂是 40–70）。`c` 全程 0.47–0.82（"看起来在接触"）。
**这就是"距离判据"漏掉的东西，也正是接触量要补的**。

#### 9.75.3 实现：bar-agnostic + load-gated 的新家族（`support_dual_load` / `support_dual_load_both`）

两个修法是**正交的、都要**：bar-agnostic 保住"B1 这块踏脚石也算数"（不跳步、且不依赖被采样的指令杆），
load-gated 补上"在窗内 ≠ 抓住"。用每只手的布尔量

```
g_h = (d[h, argmin_k d[h,k]] < 0.05)  AND  (EMA_5(F_h) > F0*theta)     # 杆无关！
```

| 变体 | goal | 维数 |
|---|---|---|
| **`support_dual_load`** | `support_dual` + `[h_any_load, h_dual_load]` | 13（state 147 / obs 160）|
| **`support_dual_load_both`** | 上面 + `[h_both_load]` | 14（state 147 / obs 161）|

* `h_any_load` / `h_dual_load` 是 `h_any` / `h_dual` 的**载荷版**（同杆双手 = `both g_h` 且在同一根杆窗内）；
* `h_both_load` = **两手都承力（可以不在同一根杆上）**。这是唯一能对**正在换手的那只手**提要求的项：
  换手过程中 `h_any_load` 已被支撑手满足、`h_dual_load` 又不说"哪只手"，只有它在说
  "你刚够到的那只手必须真的吃上力"；
* 三者**只在 `support_dual` 上做加法**，不删任何东西——run #18 已经证明删掉连续抓握项会让训练崩（§9.50），
  而距离版正是链得以形成的原因；
* 全部**不含 `k_goal`**，所以这两个变体的训练分布重新与 `--train-goal-bar` 无关（回到 §9.52 之前的好性质）。

**CPU 验收（`src/check_contact_sense.py`，19 条断言，两个变体都过）**：

| 臂 | 设置 | `[any, dual, both]` | 含义 |
|---|---|---|---|
| P | 双手闭合吊 B0 | **[0.92, 0.92, 0.92]** | 真双手承力 ⇒ 三项都满 |
| **P2** | 双手闭合、goal=**B1**（人在 B0）| **[0.92, 0.92, 0.92]** | **抓在"错杆"上照样算数** —— 这就是 bar-agnostic，B1 这块踏脚石因此能拿分 |
| H | 左手张开、右手闭合 | **[0.87, 0, 0]** | only-one-hand ⇒ `both`=0，正是"要求够到的那只手吃上力" |
| N | 松手坠落 | **[0, 0, 0]** | 无接触 |

守卫：`check_variants` 21/21 自洽、`check_metrics`/`check_args` 通过、
`check_contact_sense`（`support_dual`/`support_dual_contact`/`_lc`/`_lcb` 四组）全过。

#### 9.75.4 建议的下一跑（与 6h 基线**逐项同配方**，只改 goal 变体）

```bash
.venv-warp/bin/python -u src/train.py --preset C_l2_infonce \
  --goal-variant support_dual_load_both --goal-reach-thresh 0.35 \
  --train-goal-bar -1 --train-goal-bar-min 1 --train-goal-bar-max 2 --start-bar-max 1 \
  --eval-goal-bar 1 --scene full035 \
  --num-envs 128 --num-eval-envs 16 --episode-length 501 --batch-size 512 \
  --steps 50131712 --num-evals 60 --expl-hold 10 --save-every 5 \
  --buffer-gb 1.0 --xla-mem-fraction 0.6 --gpu 0 \
  --exp-name brach_lboth_2bar_6h --checkpoint-dir runs/ckpt_lboth_2bar_6h \
  --wandb --wandb-group .
```

判读（与 `dual2bar_6h` 对照）：`--goal-bar 2` 渲染里 (a) **先碰 B2 前是否仍有 17–19 步在 B1 上**
（bar-agnostic 是否保住了踏脚石）、(b) **B2 的 `f>0.5` 步占比是否从 0–15% 上去**、(c) 存活是否超过 57–61 步。

### 9.76 「能不能复用 6h 的权重接着跑？」——权重可以 warm start，**"接着跑"做不到**

#### 9.76.1 仓库到底存了什么（读代码 + 读文件确认，不是猜）

`runs/ckpt_contact_2bar_6h/` 只有两种产物：

| 文件 | 内容 | 大小 |
|---|---|---|
| `actor_*.pkl` / `actor_latest.pkl` | **只有 actor** 参数 + config + steps（`train.py::save_actor`）| 4.4 MB |
| `final` | **pickle 的 3 元组 `(alpha, actor, critic)`**：`{'log_alpha': -3.73}` / `{'params':…1100828}` / `{'g_encoder','sa_encoder':…2200448}` | 13 MB |

**一次真正的 resume 还需要**（全都没存，事后也无法恢复）：

* replay buffer：这里 `max_replay_size=12122 × 128 envs` ≈ **1 GiB** 的转移；
* actor / critic / alpha 各自的 **Adam 动量**（optax state）；
* RNG key、`env_steps`/`gradient_steps` 计数、以及 wrapped env 的 `pipeline_state`（轨迹 id、episode 计数）。

`CRL.train_fn` 里这些状态都是**现场 `TrainState.create(...)` 建的**（动量全 0、`env_steps=0`），
所以即使把参数塞进去，**优化过程也是从零开始**。⇒ **"接着跑"只能在下次跑之前先加全状态存档**（见 9.76.4）。

#### 9.76.2 已实现：`--init-from`（warm start，不是 resume）

* `crl.py` 加三个可选字段 `init_{alpha,actor,critic}_params`，`train_fn` 里用它**替代** `actor.init(...)` / 新建的
  `critic_params` / `log_alpha=0`；
* `train.py --init-from <path>`：接受 `runs/ckpt/*/final`（三元组）或 `actor_*.pkl`（只有 actor，则 critic/alpha 冷启）；
* **观测布局硬校验**：取 checkpoint 里 actor 的所有 2-D 层宽度，若本环境 `observation_size` 不在其中就
  **立即失败**并说明（`support_dual_contact` 159 dims ≠ `support_dual_load_both` 161 dims，**不能互换**）；
* `args.json` 与 checkpoint 的 config 里都记 `init_from`，可追溯。

实测校验（本次 6h ckpt）：actor 2-D 宽度 = **[14, 159, 256]**，159 == 本环境 obs ✓；`log_alpha = -3.73`
⇒ **alpha = 0.024**（从初始 1.0 一路自适应降下来的）。warm start 会把这个值一起带过去——
这点很关键：冷启 alpha=1.0 会让前期探索噪声大得多。

**它不是 resume 的地方**（必须记在结论里）：buffer 空、Adam 动量 0、步数从 0 计、RNG 新。
所以前 ~1 M 步会出现"critic 重新适应 + buffer 重新填充"的凹陷，**不要拿它和 6h run 的曲线逐点相接**，
只和它最后几个 eval 的数字比。

#### 9.76.3 1 h 续跑命令（先 30 秒 smoke 验加载，再正式跑）

```bash
cd ~/monkeyBars_CRL && unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES

# ① 30 秒 smoke：只验 "能不能加载 + 能不能 jit"
.venv-warp/bin/python -u src/train.py --preset C_l2_infonce --smoke \
  --goal-variant support_dual_contact --goal-reach-thresh 0.35 --train-goal-bar 1 --eval-goal-bar 1 \
  --scene full035 --expl-hold 10 \
  --init-from runs/ckpt_contact_2bar_6h/final --exp-name smoke_warm
#   看到 "[init] WARM START from …: actor input 159-D OK" 就说明加载成功

# ② 正式 1 h（≈9.1 M 步；按实测 2527 steps/s × 3600 s）
.venv-warp/bin/python -u src/train.py --preset C_l2_infonce \
  --goal-variant support_dual_contact --goal-reach-thresh 0.35 \
  --train-goal-bar -1 --train-goal-bar-min 1 --train-goal-bar-max 2 --start-bar-max 1 \
  --eval-goal-bar 1 --scene full035 \
  --num-envs 128 --num-eval-envs 16 --episode-length 501 --batch-size 512 \
  --steps 9000000 --num-evals 11 --expl-hold 10 --save-every 2 \
  --buffer-gb 1.0 --xla-mem-fraction 0.6 --gpu 0 \
  --init-from runs/ckpt_contact_2bar_6h/final \
  --exp-name brach_contact_2bar_warm1h --checkpoint-dir runs/ckpt_contact_2bar_warm1h \
  --wandb --wandb-group .
```

判读：拿 6h run 的 **eval 59**（`len 472.2 / fell 0.06 / advance_max 357.8 / bar_L 337.8 / succ 98.2`）
作基准，看 11 个 eval 之后能否超过；另外 `--goal-bar 2` 渲染里看
(a) 先碰 B2 前在 B1 的步数、(b) B2 上 `f>0.5` 的占比、(c) 存活是否超过 116–130 步。
**值不值得跑**：6h run 最后两个 eval 还在陡升（`len 243→472`、`succ 7.6→98.2`），所以 1 h 大概率能看到变化；
但它是 warm start 不是续跑，出现短暫回落是正常的。

#### 9.76.4 若以后要**真正**的"接着跑"：需要先加全状态存档

要加的东西（约 30 行 + 一条守卫）：在 `--checkpoint-dir` 里周期性 pickle
`(training_state, replay_buffer_state, env_state, rng, config)`，再加 `--resume-state`；
一份 ~1 GiB，落盘 10–30 s。**只在加了这个之后的 run 上才谈得上"接着跑"**；
本次 6h run 没有，所以无法补。

### 9.77 1 h warm start（基线 `support_dual`）：机制成立，**B1 变好、B2 变差**

`runs/brach_dual2bar_warm1h/`（`--init-from runs/ckpt_dual2bar_6h/final`，9.0 M 步 / 11 evals，
`git bd98b51`，`gpu 0`；配方与 6h 基线逐项相同，**但 eval 探针按建议改成了 `--eval-goal-bar 2`**）。

> ⚠️ **判读前提**：这 11 个 eval 是在 **B2** 指令下量的，而 6h 基线的 `len 501 / fell 0 / succ 380`
> 是在 **B1** 指令下量的 —— **两者不可比**。与基线可比的是它**在 B2 探针下的渲染**（§9.75.2 那张表）。

#### 9.77.1 机制：warm start 确实生效了

warm-start run 的 **eval 0（0.94 M 步）**就是 `len 57.8 / fell 1.00 / max_bar 50.4` —— 这正是 6h 基线
**在 B2 探针下**的行为（链到 B1 → 坠落）。从零开始的策略在 0.94 M 步时是"挂在 B0 不动"（`max_bar ≈ 0`），
所以这不是冷启。`[init]` 校验（actor 输入 155-D、alpha 0.0319 一起带过来）也过了。

#### 9.77.2 B1 探针（`--goal-bar 1`）：**没有退化，反而更准**

| | 6h 基线 @50 M | warm 1 h 后 |
|---|---|---|
| 存活 | 4/4（`-1`）| **4/4（`-1`）** |
| goal dist | 1.538 → 0.038 | 1.538 → **0.006** |
| position-only 8 维 | 0.605 → 0.026 | 0.605 → **0.006** |
| soft grasp | L 0.63 / R 0.76 | L 0.63 / **R 0.82** |
| **真实承力 `f>0.5`** | （当时无此列）| L **41%** / R **90%** 的存活步 |
| 窗内占比 | — | L 67% / R 94% |
| 手指载荷 EMA | — | L 17.1 / R **63.6** N·m |
| 换手次数 | 12/12/13/17 | 15/19/12/19 |

⇒ **B1 基本被解掉了**（右手 90% 的存活步真的抓着 B1、载荷 63.6 N·m、末端距离 0.006）。
**但注意 L 的 `hover` 是 20%**：左手有 20% 的存活步"在 B1 窗内但不承力"——这正是
`support_dual_load_both` 的 `h_both_load` 会判 0 的那个状态（§9.75.3），也是 bar-agnostic 距离判据的遗留漏洞。

#### 9.77.3 B2 探针（`--goal-bar 2`）：**变差了**

| | 6h 基线 @50 M | warm 1 h 后 |
|---|---|---|
| 坠落 | 4/4 @ t = 57/60/61/58 | 4/4 @ t = **62 / 27 / 27 / 68** |
| 到 B2 最近距离 | L **0.001–0.028**（4/4 都到）| L 0.016 / **0.523** / **0.525** / 0.003（**只有 2/4 到**）|
| B2 窗内（L）| 25–33% | **0–21%** |
| B2 真承力（L）| 0–15% | **0–4%** |
| B2 悬停（L）| 0–9 步 | 0–3 步 |
| 载荷去向 | L 31.8 / R 45.9 N·m（都在**身后**的杆上）| L 37.6 / R 37.7 N·m（同）|

⇒ 1 h warm start **没有解决 B2，反而让"链到 B1 就守住"这个解更占优**：两个 episode 现在在 t=27 就坠落，
左手的座连 B2 的 5 cm 都进不去（0.52 m）。这与"B1 变得更漂亮"是同一件事的两面。

**机制解释（假设）**：课程里 25% 是 B0→B1（"守住 B1"就完全解掉）、25% 是 B1→B2（同样以守 B1 为前提），
而 B2 从来没有成功过一次 ⇒ 梯度把权重全推给了"到 B1、抓住、守住"这个**可达且被奖励**的解，
B2 的尝试被剪掉。这与 §9.73/#21 的后期回退是同一类"吸引子塌缩"。

#### 9.77.4 结论与下一步

* **"再跑久一点"不是瓶颈**：与 §9.75.1 的 checkpoint 扫描一致（16→50 M 步到达变好、守住没变）。
  本次 warm start 又给了一次独立证据：B1 继续变好，B2 继续不动/变差。
* **warm start 这条工具是好用的**（机制已验证、B1 无损），所以下一步应当**用它去改课程，而不是延长同一课程**：
  最便宜的 1 h 实验是把课程里的 B0→B1 那一档去掉、只留"必须以 B2 为目标"的两种 episode：

```bash
.venv-warp/bin/python -u src/train.py --preset C_l2_infonce \
  --goal-variant support_dual --goal-reach-thresh 0.35 \
  --train-goal-bar -1 --train-goal-bar-min 2 --train-goal-bar-max 2 --start-bar-max 1 \
  --eval-goal-bar 2 --scene full035 \
  --num-envs 128 --num-eval-envs 16 --episode-length 501 --batch-size 512 \
  --steps 9000000 --num-evals 11 --expl-hold 10 --save-every 2 \
  --buffer-gb 1.0 --xla-mem-fraction 0.6 --gpu 0 \
  --init-from runs/ckpt_dual2bar_warm1h/final \
  --exp-name brach_dual2bar_b2only1h --checkpoint-dir runs/ckpt_dual2bar_b2only1h \
  --wandb --wandb-group .
```

（`gmin=gmax=2` + `start_bar_max=1` 的采样表恰好给出 **(B1→B2) 50% / (B0→B2) 50%**，没有 B0→B1；
观测布局不变（155），所以**可以继续 warm start**。目标看 `advance_max` 是否逼近 2、
B2 的 `f>0.5` 占比是否从 0–15% 抬起来、坠落时间是否超过 68 步。）

* 另一条路是 `support_dual_load_both`（§9.75.3）：它针对的正是上面观测到的两个漏洞
  （B2 窗内不承力、B1 上左手 20% 悬停），但它的 obs 是 **161** 维、**不能 warm start**，需冷启 6h。

### 9.78 B2-only 课程（1 h warm start）：**B2 的"接近"变稳了，但承力仍然没有；B1 被牺牲**

`runs/brach_dual2bar_b2only1h/`（`--init-from runs/ckpt_dual2bar_warm1h/final`，
`--train-goal-bar-min 2 --train-goal-bar-max 2 --start-bar-max 1` ⇒ 采样恰好是 **(B1→B2) 50% / (B0→B2) 50%**，
**没有任何 B0→B1**；`--eval-goal-bar 2`，9.0 M 步 / 11 evals，`git 25b05e3`）。

#### 9.78.1 eval 曲线（B2 探针）

`advance_max` 比上一轮改善：0（起点）→ **16.7**（eval 10），峰值 23.1 @eval 1；
`max_bar` 55→63，`len` 峰值 **125.2**。但 `fell` 恒 **1.00**、`succ` 恒 0，
B2 上真承力步数 `ctL` 只有 5.9–10.6（每 episode 约 70 步 ⇒ **8–15%**）。
注意 `nrL ≈ ctL`（5.9/5.9、10.6/11.1、7.5/7.5）⇒ **一进 B2 的窗就承力，几乎没有悬停**。

#### 9.78.2 三个 run 的 B2 探针逐 episode 对照（存活期）

| | 基线 6h @50M | warm1h（混合课程）| **b2only1h** |
|---|---|---|---|
| 存活步数 | 57/60/61/58 | 62/**27**/**27**/68 | **62/65/75/72** |
| B2 窗内步数 | 14/20/15/15 | 3/0/0/14 | 3/3/13/15 |
| **B2 真承力步数** | **0/5/9/5** | 0/0/0/3 | 0/0/1/4 |
| 到 B2 最近距离 | 0.023/0.001/0.028/0.003 | 0.016/**0.523**/**0.525**/0.003 | 0.045/0.022/0.013/0.005 |
| B2 悬停 | 0–9 步 | 0–3 步 | **0 步** |

⇒ B2-only 课程买到了**稳定性**：4/4 都进 B2 的 5 cm 窗（上一轮只有 2/4）、4/4 都活到 62–75 步、
且完全没有"贴杆不抓"。但**真承力仍只有 0–4 步** —— 而**基线反而是三者里承力最多的（0/5/9/5 步）**。
也就是说：两次 1 h warm start 在"接近"上变稳了，在**"把载荷交过去"这件事上都没有前进（甚至更差）**。

#### 9.78.3 B1 探针：**被明显牺牲**（这是 B2-only 课程的代价）

| | 基线 6h @50M | warm1h | **b2only1h** |
|---|---|---|---|
| 存活 | 4/4 | 4/4 | 4/4 |
| goal dist | 0.038 | **0.006** | **0.780** |
| position-only 8 维 | 0.026 | 0.006 | 0.270 |
| 右手真实承力 `f>0.5` | — | **90%** | **16%** |
| B1 窗内占比 L/R | — | 67% / 94% | **1% / 18%** |
| 换手次数 | 12/12/13/17 | 15/19/12/19 | **45/47/53/33** |
| 末端躯干 x | +0.347 m | +0.334 m | **+0.083 m**（没停在 B1 下）|

⇒ 它**没有坠落**，但"挂在 B1 下"这个解被**忘掉了**：右手只有 16–18% 的时间在 B1 上，
躯干停在 B0 附近、换手次数翻 3 倍。因为课程里已经**没有"守住 B1 就完全解掉"的 episode**，
B1 只剩工具价值，于是策略学会了"多摆少守"。

#### 9.78.4 结论：B2 的瓶颈既不是训练时长，也不是课程配比

三次 1 h（同一 6 h 基线 warm start）合起来给出一个**取舍前沿**而不是进展：
**B1 解掉（dist 0.006）↔ B2 接近变稳（4/4 进窗、活到 75 步）**，两者互斥；
而"**把载荷交给 B2**"在三次里都没发生（真承力 0–9 步，最好的一次还是**原始基线**）。

⇒ 剩下的候选只有**改 goal** 那一类，而且 §9.75.3 的 `support_dual_load_both` 正对着这次测到的两个漏洞
（B2 窗内不承力 / 到达后不接载荷）。它的观测是 **161** 维、与 155 维不兼容，所以只能：
**(a)** 冷启 6 h 直接跑 `support_dual_load_both`；或
**(b)** 我加一个 `--init-from-pad`（把新增的 3 个 state + 3 个 goal 输入维**用 0 补齐到新宽度**，
    保留原有 155 维的权重，即"从一个已解 B1 的策略出发"），再用 1 h warm start 试。
