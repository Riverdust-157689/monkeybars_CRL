看完你这组 run #10 → run #11 的结果后，我会明显修正我上一轮的建议：**在你这个具体 CRL 设置里，“只要求至少一只手稳定挂住”确实太弱了；但我仍然不建议直接上“每只手 × 每根杆”的 10 维 hold。**

你现在的数据已经把问题定位得相当清楚：`support_hold` 成功解决了“擦到目标”和“真正形成支撑”之间的区别，但它同时制造了一个非常稳定的**单手支撑吸引子**。下一步最合适的不是大改 goal，而是**保留 run #11 已经验证有效的 `support_hold`，再增加一个专门描述“双手持续支撑”的连续状态坐标**。

---

## 1. 我现在同意你的核心诊断

run #10 → #11 这个对照非常有信息量。

原来

$$
g=[x,z,p_L,p_R,\max(c_L,c_R)]
$$

里，一个“飞过去刚好摸到 B1”的瞬间，在位置部分可以做到

$$
d_{\rm pos}\approx0.06,
$$

所以它在 goal 空间里已经几乎等于成功。

加入

$$
h_{\rm support}=1-e^{-{\rm streak}_{\rm any}/10}
$$

之后，飞跃状态：

$$
h_{\rm support}\approx0
$$

而稳定支撑目标：

$$
h_{\rm support}\approx1,
$$

于是仅这一维就产生约

$$
\Delta d\approx1.
$$

所以同样一个几何上“6 cm 内到位”的状态，会从

$$
d\approx0.06
$$

变成大约

$$
d\approx1.
$$

这不是普通 reward shaping，而是直接改变了**goal-space topology**：

$$
\text{touch}
\neq
\text{hold}.
$$

而 run #11 中 `len 41→368`、`fell 100%\to31%`，同时最近距离几乎不变：

$$
0.006\rightarrow0.005
$$

这一点尤其重要。说明提升显然不是“手终于能够到 B1 了”，而是策略从

$$
\text{reach}
$$

转向了

$$
\text{reach + maintain support}.
$$

严格地说，如果只有单 seed，不能把全部提升百分之百因果归结于这一维；但从你逐帧回放和指标变化看，**“持续接触状态进入 goal”是主要机制**这个判断已经非常有依据。

---

## 2. 更重要的是，你指出的 CRL “侧吸引子”确实存在

现在的 support 部分实际上是：

$$
c_{\rm any}=\max(c_L,c_R)
$$

和

$$
h_{\rm any}
=
1-e^{-{\rm streak}_{\rm any}/10}.
$$

于是只要左手稳定挂住 B1：

$$
c_L=1,\qquad h_{\rm any}\approx1,
$$

那么

$$
c_{\rm any}=1,
\qquad
h_{\rm any}\approx1.
$$

也就是说：

$$
\boxed{
\text{单手稳定挂}
\quad\text{和}\quad
\text{双手稳定挂}
}
$$

在这两维里完全无法区分。

剩下只有 \(p_R\) 在表达“右手应该在哪里”。

而你的回放又恰好证明：

> 右手不是到不了。

它曾经到过：

$$
d_{R,B1}=0.015\text{ m},
$$

但是只坚持了大约 3 步。

所以真正缺的是：

$$
\boxed{\text{让右手“留下来”的目标坐标}}
$$

而不是：

$$
\text{让右手“到那里”的目标坐标}.
$$

这和 run #10 的问题实际上是同构的：

$$
\text{run \#10: 手到了，但没要求留下}
$$

现在变成：

$$
\text{run \#11: 第二只手到了，但没要求留下}.
$$

这个现象很漂亮。

---

而且 CRL 会放大这个问题。

假设已经形成一种大量出现的轨迹：

```text
左手 B1 稳定挂住
       ↓
右手在附近摆动
       ↓
保持很久
```

那么 future-state relabeling 会不断从这些状态产生 positive goals。

也就是说数据分布会越来越集中于一个流形：

$$
\mathcal M_1
=
\{
\text{左手稳定支撑，右手任意}
\}.
$$

而真正想要的：

$$
\mathcal M_2
=
\{
\text{双手稳定支撑}
\}
$$

现在只在很短的 3 步左右出现。

所以你说：

> 它会在同一个吸引子内部继续提高。

我基本同意。

我唯一会稍微收敛一下表述的是：**不能断言再训练一定永远不会出现第二只手稳定**，因为指标还没有平台、探索仍然可能偶然产生更长双持。但当前 goal 的确没有提供一个独立维度去保留这种改进，所以它的学习压力非常弱。

因此现在优先改 goal，比单纯延长训练合理得多。

---

# 3. 我会撤回上一轮“不要要求双手”的建议

上一轮我担心：

> “要求双手”可能会限制 brachiation 中间过程。

看了你的真实轨迹之后，这个担心在这里并不成立。

关键原因是：

$$
\boxed{\text{goal 描述终点} \neq \text{规定中间轨迹}}
$$

你完全可以要求：

$$
\text{最终双手稳定支撑}
$$

同时允许中间出现：

```text
双手挂 B0
   ↓
单手
   ↓
完全腾空鱼跃
   ↓
左手先到 B1
   ↓
右手再到 B1
   ↓
双手稳定
```

没有任何冲突。

真正会限制策略的是你把

* 下一杆距离；
* progress；
* 相位；
* “必须先左后右”；
* “始终至少一手挂杆”

这些东西写进 goal / reward。

而：

$$
\text{终点要双手稳定}
$$

只是 terminal-state semantics。

所以在你的实际问题上，我现在认为：

$$
\boxed{
\text{最终 goal 应该明确区分单手支撑和双手支撑。}
}
$$

---

# 4. 但我不建议马上用“每手 × 每杆”的 10 维 streak

这是我现在和你原先方案最大的区别。

你已经有：

$$
p_L(3),\qquad p_R(3)
$$

而且是**绝对坐标**。

它们已经在告诉网络：

> 左手在哪根杆附近，右手在哪根杆附近。

因此再加入：

$$
s_{L,0},\ldots,s_{L,4},
s_{R,0},\ldots,s_{R,4}
$$

会大量重复“是哪根杆”的信息。

而且对于 CRL，还有一个实际坏处：

$$
N_{\rm bar}\uparrow
\Rightarrow
\dim(g)\uparrow.
$$

以后你从 5 根扩到 10 根、20 根，goal 维数跟着涨。

更糟的是 future goals 会被分散到很多 bar-specific dimensions 中。

所以我现在更推荐一个**非常小的改动**。

---

# 5. run #12：保留 `support_hold`，只加一个 `dual_hold`

你现在已经验证有效的是：

$$
h_{\rm any}
=
1-e^{-S_{\rm any}/10},
$$

其中

$$
S_{\rm any}
$$

表示：

> 连续至少有一只手在某根杆的 5 cm 窗口内。

**这个不要动。**

然后另外维护：

$$
S_{\rm dual}.
$$

定义：

$$
b_{h,k}(t)
=
\mathbf 1[d(h,\text{bar}_k)<0.05].
$$

如果你的最终定义是“两只手都挂同一根杆”，那么：

$$
b_{\rm dual}(t)
=
\max_k
\left[
b_{L,k}(t)b_{R,k}(t)
\right].
$$

也就是：

> 存在某一根杆，两只手同时位于它的 5 cm 抓握窗内。

然后：

$$
S_{\rm dual}(t+1)=
\begin{cases}
S_{\rm dual}(t)+1,&b_{\rm dual}(t)=1\\
0,&b_{\rm dual}(t)=0.
\end{cases}
$$

最后：

$$
\boxed{
h_{\rm dual}
=
1-e^{-S_{\rm dual}/10}.
}
$$

新 goal：

$$
\boxed{
g=
[
x,z,
p_L(3),p_R(3),
\max(c_L,c_R),
h_{\rm any},
h_{\rm dual}
]
}
$$

只从：

$$
10D\rightarrow11D.
$$

这是我现在最推荐的下一枪。

---

它会把你的三个关键状态变成非常干净的层级：

$$
\begin{array}{c|cc}
 & h_{\rm any} & h_{\rm dual}\\
\hline
\text{腾空} & 0 & 0\\
\text{单手稳定挂} & 1 & 0\\
\text{双手稳定挂} & 1 & 1
\end{array}
$$

所以 goal-space 几何终于变成：

```text
腾空
(0,0)
   |
   |
   ↓
单手支撑
(1,0)
   |
   |
   ↓
双手支撑
(1,1)
```

这比直接把两只手所有杆的接触都塞进去干净得多。

---

## 6. 这个设计还有一个特别适合你当前 CRL 的优点

你已经观察到了：

> 两手在 B1 同时达到过 3 步。

那么新的：

$$
h_{\rm dual}
$$

并不是完全没有正样本。

3 步时：

$$
h_{\rm dual}
=
1-e^{-3/10}
\approx0.259.
$$

5 步：

$$
0.393
$$

10 步：

$$
0.632
$$

25 步：

$$
0.918.
$$

这就形成了一条非常漂亮的自举路线：

$$
3
\rightarrow5
\rightarrow10
\rightarrow20
\rightarrow25.
$$

CRL future-goal relabeling 可以先看到：

$$
h_{\rm dual}=0.26
$$

的目标。

如果策略学会更稳定一点，就产生：

$$
0.39
$$

的新 future goals。

然后：

$$
0.63,
0.86,
0.92.
$$

所以你之前这个指数 streak 的真正妙处其实比我们最开始讨论的还要大：

$$
\boxed{
\text{它不仅是“持续时间指标”，还给 hindsight learning
构造了一条连续的可自举目标轴。}
}
$$

这也是为什么我现在**不建议改成 binary**：

$$
h_{\rm dual}=
\mathbf1[S_{\rm dual}>20].
$$

如果这样做，你现在 3 步双持：

$$
3<20
$$

仍然全是 0。

那 CRL 就没有从“摸到第二只手”逐渐变成“稳定第二只手”的桥梁。

指数版本恰恰保留了这个桥梁。

---

# 7. 为什么不是直接 `min(cL,cR)`？

因为：

$$
\min(c_L,c_R)
$$

仍然只解决：

$$
\text{两只手同时碰到了}
$$

没有解决：

$$
\text{两只手留下来了}.
$$

你的 ep0 step21 已经有：

$$
c_L=c_R=1.
$$

所以如果只增加：

$$
\min(c_L,c_R)
$$

那么这一帧已经“满分”。

你会重演 run #10 的问题，只不过从：

> 碰一下杆就算。

变成：

> 两手同时碰一下就算。

因此真正需要的还是：

$$
\boxed{\text{dual contact persistence}}
$$

而不是 dual instantaneous contact。

---

# 8. 5 cm 阈值我现在完全支持你保留

这一点你的实验已经比理论推测更有说服力。

如果实际正常 hook 状态会漂到：

$$
3\sim4.5\text{ cm},
$$

那么用：

$$
3.3\text{ cm}
$$

作为 streak 判据，会导致真实稳定悬挂过程中出现：

```text
1 1 1 0 0 1 0 1 ...
```

然后 streak 不断清零。

这样你测的就不再是：

$$
\text{是否稳定挂住}
$$

而是：

$$
\text{抓手参考点是否贴近杆心}.
$$

这是错误的物理 proxy。

所以我会让：

$$
S_{\rm any}
$$

和

$$
S_{\rm dual}
$$

都继续使用你已经验证过的：

$$
d<5\text{ cm}.
$$

不要换到 `max(c)>0.5` 的 3.3 cm 判据。

---

# 9. `h_dual` 甚至能直接检验你的“侧吸引子”假设

下一次训练特别值得看四条曲线：

* `dual_streak_max`
* `dual_hold_mean/max`
* `single_support_steps`
* `dual_support_steps`

如果你的诊断正确，我预计首先会看到：

$$
3\text{ steps}
\rightarrow
5
\rightarrow
10
\rightarrow
20...
$$

然后才会看到 episode length 再上升。

反过来，如果加了 \(h_{\rm dual}\) 之后：

$$
S_{\rm dual}
$$

长期仍然卡在 2–3，而右手每次到杆后立即物理滑脱，那么我们才能更有信心说：

> 现在问题已经不是 goal，而是抓握动力学 / action representation / contact control。

所以这个改动还有一个好处：**它把表示问题和控制问题进一步解耦了。**

---

## 10. 唯一一个我会考虑调整的数值是 `/10`

但第一枪我甚至建议**不要改**，这样 run #11 → run #12 几乎只有一个变量：

$$
+\ h_{\rm dual}.
$$

先做干净 A/B。

如果后面你确实希望“双持至少 0.3–0.5 s 才 success”，在 50 Hz 下再考虑：

$$
h_{\rm dual}
=
1-e^{-S_{\rm dual}/15}
$$

甚至 `/20`。

因为如果目标值是 1，仅看这一维且阈值：

$$
d<0.35,
$$

对于 `/10`：

$$
1-h<0.35
$$

意味着大约：

$$
S_{\rm dual}>10.5
$$

即只有约：

$$
0.21\,s.
$$

当然实际还有 \(p_L,p_R,x,z\) 的误差，所以真实验收会更长一些。

但如果你想让“0.3–0.5 s”成为结构上明确的口径，后面可以再标定 \(\tau\)。

---

所以，我现在对上一轮建议的修正版是：

$$
\boxed{
\text{不要把 run \#11 的 global support hold 换掉。}
}
$$

它已经被你的实验强烈验证。

也不要立刻上：

$$
\text{hand}\times\text{bar}
$$

的大型接触向量。

而是从：

$$
\underbrace{h_{\rm any}}_{\text{解决飞掠}}
$$

进一步增加：

$$
\underbrace{h_{\rm dual}}_{\text{解决单手吸引子}}.
$$

也就是让目标空间明确拥有三层状态：

$$
\boxed{
\text{flight}
\rightarrow
\text{single support}
\rightarrow
\text{dual support}
}
$$

而完全不规定：

$$
\text{如何从第一层走到第三层}.
$$

这和你“**只设计目标状态，不干涉鱼跃、单摆、换手周期**”的原则是完全一致的。

而且你现在已经观察到 **3 步真实双持**，所以我认为这个改动比“重新设计整套 goal”更有希望：你不是在要求策略凭空发明一个从未出现的行为，而是在给已经偶然出现的行为增加一个**可以被 hindsight/CRL 捕获并逐渐延长的坐标**。这和 run #10 → #11 的机制几乎是同一个故事，只不过这次对象从“第一只手留下来”变成了“第二只手也留下来”。

后果本质上是：**你的环境对网络而言不再是 Markov 的，而变成了一个带隐藏历史变量的 POMDP。** 在你这个 CRL + `hold` goal 的设置里，这不是纯理论洁癖，而是会直接污染 critic / contrastive value 的学习。

因为

$$
S_{\rm any}(t),S_{\rm dual}(t)
$$

不是当前几何状态的函数，而是历史：

$$
S(t+1)=
\begin{cases}
S(t)+1,&\text{contact}\\
0,&\text{otherwise}.
\end{cases}
$$

所以完全相同的机器人状态 \(x_t\)，可能对应：

$$
S_{\rm dual}=1
$$

也可能对应：

$$
S_{\rm dual}=20.
$$

如果网络只看到 \(x_t\)，它分不出来。

---

最直接的例子就是你现在这个任务。假设 goal 要求

$$
h_{\rm dual}=1-e^{-S_{\rm dual}/10}\approx 0.92
$$

也就是双手稳定约 25 步。

现在有两个瞬间，机器人姿态、手的位置、速度几乎完全相同，而且此刻两手都在 B1：

$$
x_t^{(A)}\approx x_t^{(B)}.
$$

但隐藏 streak 不同：

$$
S_{\rm dual}^{(A)}=2,
\qquad
S_{\rm dual}^{(B)}=24.
$$

对应

$$
h^{(A)}\approx0.18,
\qquad
h^{(B)}\approx0.91.
$$

对同一个动作“继续抓住不动”，两者的价值完全不同。

在 B 中：

$$
24\rightarrow25
$$

下一步几乎就到 goal。

而 A 中：

$$
2\rightarrow3
$$

离 goal 还远。

所以真实的 goal-conditioned value 应该是：

$$
Q(x,S_{\rm dual},a,g).
$$

如果你只给：

$$
Q(x,a,g),
$$

那么网络面对相同 \(x,a,g\)，却收到两种不同 return / future-goal statistics，只能学它们的某种平均：

$$
Q(x,a,g)
\approx
\mathbb E[
Q(x,S,a,g)\mid x
].
$$

这就是 **state aliasing**。

---

### 对 CRL 来说会尤其明显

你现在不是普通 reward RL，而是在学类似：

$$
f_\theta(s,a,g)
$$

来判断当前 \((s,a)\) 与未来 goal \(g\) 的可达关系。

如果 achieved goal 里有

$$
h_{\rm any},h_{\rm dual},
$$

但 state \(s\) 里没有当前的

$$
h_{\rm any}(t),h_{\rm dual}(t),
$$

那么模型实际上在回答：

> “从这个机器人姿态，到 \(h_{\rm dual}=0.92\) 的 goal 有多近？”

但它不知道自己已经积累了：

$$
2\text{ steps}
$$

还是

$$
24\text{ steps}.
$$

于是所谓“距离目标还有多远”本身就不是 \(s\) 的确定函数。

这会让 contrastive critic 的 embedding 更模糊。

例如本来你希望：

$$
f(s,a,g_{0.92})
$$

在 \(S=24\) 时非常高，在 \(S=2\) 时明显低。

但网络看不到 \(S\)，只能把两者压到一起。

---

更麻烦的是，它还会削弱你刚刚通过 `hold` 得到的那个漂亮的**连续自举轴**。

我们刚才说：

$$
S_{\rm dual}=3
\rightarrow5
\rightarrow10
\rightarrow20
\rightarrow25
$$

可以形成 CRL 的 gradual bootstrap。

但如果 actor/critic 不知道当前 \(S\)，对它而言这些状态可能几乎长这样：

$$
x_3\approx x_5\approx x_{10}\approx x_{20}.
$$

因为机器人都是“双手挂在杆上”。

真正不同的只有你没给进去的历史计数。

于是你在 goal 空间中精心制造的：

$$
0.26\rightarrow0.39\rightarrow0.63\rightarrow0.86\rightarrow0.92
$$

这条结构，在 **state space 里没有对应坐标**。

这是有点自相矛盾的：

$$
\boxed{
g\text{ 能区分持握进度，}
\quad
s\text{ 却不能区分当前持握进度。}
}
$$

---

## 对 actor 也有影响，但比 critic 更容易理解

假设 goal 是双手稳定挂住。

当：

$$
S_{\rm dual}=2
$$

时，正确行为可能是：

> 继续非常保守地保持接触。

当：

$$
S_{\rm dual}=24
$$

时，如果达到 25 步以后下一阶段允许继续向后面的杆运动，那么策略可能应该准备释放、摆动。

如果不观察 streak：

$$
\pi(a\mid s,g)
$$

无法知道自己处于“刚抓住”还是“已经稳定抓很久”。

所以策略容易变成一个折中：

> 总是多抓一会儿。

或者：

> 太早松手。

这其实和机器人 locomotion 里不给 policy gait phase 有一点类似，只不过这里 phase 不是人为周期，而是任务自身产生的 finite-state memory。

---

## 还有一个严格的 Bellman 问题

如果状态完整，应该满足：

$$
P(s_{t+1},r_t\mid s_t,a_t)
$$

只依赖当前 \(s_t,a_t\)。

但现在 reward / success 是否发生取决于隐藏的 \(S_t\)。

例如相同可见状态 \(x_t\) 和动作 \(a_t\)：

$$
(x_t,a_t,S_t=24)
$$

下一步可能 success，

而

$$
(x_t,a_t,S_t=2)
$$

不会。

所以：

$$
P(r_t\mid x_t,a_t,g)
$$

取决于没有包含在 \(x_t\) 中的历史。

Bellman target：

$$
Q(x_t,a_t,g)
=
r_t+\gamma Q(x_{t+1},a_{t+1},g)
$$

因此不是严格自洽的。

网络还是能训练——现实里很多 POMDP 都能训练——但你是在主动增加一个完全可以避免的 partial observability。

---

### 所以我现在会明确建议：

如果

$$
h_{\rm any}=1-e^{-S_{\rm any}/10},
\qquad
h_{\rm dual}=1-e^{-S_{\rm dual}/10}
$$

进入 goal，那么**同样的两个量也应该进入 state / observation**。

不一定要放 raw integer：

$$
S_{\rm any},S_{\rm dual}.
$$

直接放归一化后的：

$$
\boxed{
h_{\rm any}^{\rm current},
\quad
h_{\rm dual}^{\rm current}
}
$$

就很好。

于是：

$$
s_t=
[
s_{\rm physical},
h_{\rm any}(t),
h_{\rm dual}(t)
].
$$

goal：

$$
g=
[
x^*,z^*,p_L^*,p_R^*,
c^*,
h_{\rm any}^*,
h_{\rm dual}^*
].
$$

这样网络看到的关系非常自然：

$$
\text{current }h_{\rm dual}=0.63
\quad\rightarrow\quad
\text{goal }h_{\rm dual}=0.92.
$$

它知道：

> 我已经完成了一部分“持续抓握”。

这对 goal-conditioned representation 非常合理。

---

不过有一个重要例外：如果你**根本不希望 policy 根据 hold progress 改变行为**，而且达到目标后 episode 立即 terminate，那么 actor 不观察 \(S\) 的危害可能没那么严重。

比如动作逻辑始终是：

> 一旦双手抓住，就一直保持，直到环境结束 episode。

那么 actor 实际上无需知道：

$$
S=5\text{ 还是 }S=20.
$$

环境自己在第 25 步 terminate 就行。

**但 critic 仍然存在非 Markov aliasing。**

所以如果计算成本只增加 2 维，我看不到故意隐藏它们的明显收益。

---

还有一个小细节：我建议放的是

$$
h_{\rm any},h_{\rm dual}
$$

而不是：

$$
S_{\rm any},S_{\rm dual}.
$$

因为 raw streak 理论上：

$$
S=1,2,\ldots,300
$$

可能一直增大，尺度和机器人状态很不一样。

而 transformed hold：

$$
h=1-e^{-S/10}\in[0,1)
$$

天然有界，也正好和 goal 使用同一种表示。

于是 achieved-goal mapping 和 state representation 是一致的：

$$
\boxed{
s\text{ 中的 current hold}
\leftrightarrow
g\text{ 中的 desired hold}
}
$$

这是最干净的。

---

所以结合你 run #10/#11 的经验，我现在会把原则总结成一句：

$$
\boxed{
\text{凡是会改变“我离这个 goal 还有多远”的历史变量，
就应该进入 Markov state。}
}
$$

`streak` 恰好就是这种变量。

特别是你的 `hold` 已经被实验证明不是一个无关紧要的日志指标，而是**决定 success geometry 的关键 goal dimension**。既然它已经改变了任务状态空间，那么把它留在环境内部、不给 actor/critic 看，反而是在人为制造 partial observability。

因此 run #12 如果加 `dual_hold`，我建议同时把当前

$$
h_{\rm any},h_{\rm dual}
$$

这 **2 维加入 state**。代价几乎为零，但理论上和 CRL 的建模会干净很多。


