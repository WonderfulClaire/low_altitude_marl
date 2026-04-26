# 第四章最终写作策略：保留 baseline vs fixed reward shaping

## 1. 最终实验主线

根据当前复跑结果，标准 MAPPO baseline 在 300k frames 下的 mean return 约为 0.963，与原论文中记录的 baseline 结果一致。此前 APF-AW-MAPPO 的结果明显低于 baseline，因此不建议将 APF-AW 作为正文主实验。

最终第四章建议仅保留以下两组主实验：

| 方法 | 定位 | 结果口径 |
|---|---|---|
| MAPPO baseline | 强基线，直接优化 VMAS/navigation 默认奖励 | episode reward mean ≈ 0.963 |
| MAPPO + fixed reward shaping | 本文主方法，引入进度、安全和平滑约束 | episode reward mean ≈ 0.883 |

APF-AW-MAPPO 可以不写入正文。如果需要保留，可仅作为附录或实验探索记录，不作为论文核心贡献。

## 2. 核心结论

本文不应写成“reward shaping 方法在最终回报上超过 baseline”，而应写成：

> 在当前训练预算与超参数设置下，标准 MAPPO baseline 在原始 episode reward mean 上表现更优；但 reward shaping 方法通过引入进度奖励、安全惩罚和平滑惩罚，显式改变了策略优化目标，使训练过程能够体现低空多无人机路径规划中对目标推进、安全裕度与动作连续性的建模需求。

这个结论更稳妥，也更符合实际实验结果。

## 3. 第四章推荐结构

### 4.1 实验环境与参数设置

说明 VMAS/navigation、BenchMARL、MAPPO、训练预算 300k frames、随机种子 seed=0、网络结构 MLP、50 次迭代等。

### 4.2 对比方法与评价指标

介绍 MAPPO baseline 和 MAPPO + fixed reward shaping。评价指标包括 episode reward mean、reward mean、loss 曲线、agent collisions、pos_rew、final_rew 等。

### 4.3 训练回报对比实验

报告 baseline 0.963、reward shaping 0.883。强调 baseline 在原始默认奖励上更高。

### 4.4 奖励塑形对学习动态的影响

分析 reward shaping 初期较低的原因：安全惩罚和平滑惩罚在随机探索阶段容易被触发，拉低早期累计回报；但进度奖励可以为策略提供更稠密的过程反馈，使其逐步追赶。

### 4.5 多目标权衡与实验讨论

讨论 task reward、safety、smoothness 之间的 trade-off。说明低空路径规划不能只看单一 episode reward。

### 4.6 本章小结

总结 baseline 更优、reward shaping 未超过 baseline，但验证了行为约束型奖励设计的可行性和局限性。

## 4. 建议正文表述

> 实验结果显示，标准 MAPPO baseline 在 300k frames 训练预算下取得了 0.963 的 episode reward mean，而加入奖励塑形后的方法取得了 0.883。由此可见，在当前参数设置下，reward shaping 并未进一步提高环境默认回报。其原因在于 baseline 的优化目标与 VMAS/navigation 默认评价指标完全一致，而 reward shaping 方法额外引入了进度、安全和平滑约束，使策略优化目标从单一任务回报转向多目标行为约束。因此，本文方法的价值不应仅从最终 reward 数值判断，而应结合低空多无人机路径规划任务中对安全裕度、控制连续性和策略可解释性的需求进行综合分析。

## 5. 答辩解释口径

如果老师问“为什么改进方法没有超过 baseline”，建议回答：

> 本文的实验结果没有回避这个问题。baseline 直接优化环境默认 reward，因此在该指标上表现更好是合理的。本文 reward shaping 的目标不是简单刷高默认 reward，而是把路径规划中的安全裕度和动作平滑性显式写入训练目标。实验说明固定权重 reward shaping 会改变学习轨迹，但也会带来任务回报与行为约束之间的权衡。这一结果反而说明本文实验是可复现、可解释的，也为后续采用自适应权重、多目标强化学习或多随机种子统计分析提供了方向。
