# 第四章实验评估策略：从单一回报转向多指标路径规划评估

## 1. 当前实验现象

在统一使用 VMAS/navigation 与 MAPPO 框架的前提下，标准 MAPPO baseline 往往能够在原始环境奖励上获得更高的 episode reward mean。这一现象并不意外，因为 baseline 直接优化 VMAS/navigation 的默认奖励函数，其训练目标与日志中的主要回报指标完全一致。

相比之下，reward shaping 方法在原始任务奖励之外引入了进度、安全和平滑等附加项，实际优化目标已经从单一任务回报变为多目标行为约束。因此，如果仍然只以 episode reward mean 作为唯一指标，容易低估 reward shaping 对路径规划行为质量的作用。

## 2. 论文中的核心解释角度

本文不将 reward shaping 简单表述为“必然提高原始任务回报”的技巧，而是将其定位为：

> 面向低空多无人机协同路径规划任务的行为约束机制，用于在任务完成之外进一步引导策略关注目标推进、安全裕度与控制平滑性。

因此，第四章实验分析的重点应从“reward 是否最高”转向“路径规划综合质量是否更符合低空飞行需求”。

## 3. 建议保留的实验组

建议第四章保留三组主实验：

| 实验组 | 方法 | 作用 |
|---|---|---|
| A | MAPPO baseline | 原始 VMAS/navigation 奖励下的强基线 |
| B | MAPPO + fixed reward shaping | 本文主要改进版本，结果与 baseline 差距较小，适合作为主方法 |
| C | APF-AW-MAPPO | 进一步扩展的自适应多势场版本，可作为拓展实验或参数敏感性分析 |

其中 B 作为论文主方法更稳，因为它与 baseline 的 raw reward 差距较小；C 可以作为“更强约束未必带来更高原始回报”的补充分析。

## 4. 推荐评价指标体系

第四章不要只报告 episode reward mean。建议采用以下指标体系：

| 指标类别 | 指标名称 | 解释 |
|---|---|---|
| 任务完成能力 | episode_reward_mean | 原始任务回报，衡量整体任务表现 |
| 目标推进能力 | pos_rew | 衡量智能体向目标点接近的程度 |
| 终点完成能力 | final_rew | 衡量是否获得终点相关奖励 |
| 安全性 | agent_collisions | 衡量智能体间碰撞或碰撞惩罚代理指标 |
| 训练稳定性 | reward curve volatility | 衡量训练曲线波动程度 |
| 收敛效率 | AUC / first positive step | 衡量整体训练过程表现 |

## 5. 结果解释模板

如果 baseline 的 episode reward mean 高于改进方法，可以写成：

> 从原始 episode reward mean 看，标准 MAPPO baseline 由于直接优化 VMAS/navigation 默认奖励函数，因此在该指标上具有优势。加入 reward shaping 后，策略优化目标中额外包含安全裕度与轨迹平滑性偏好，使得原始任务回报并未进一步提升。这说明在多无人机协同导航任务中，单纯追求环境默认回报与引入行为约束之间存在一定权衡。

如果 fixed reward shaping 与 baseline 差距较小，可以写成：

> 尽管 fixed reward shaping 方法在最终 episode reward mean 上略低于 baseline，但二者差距较小，说明引入进度、安全和平滑约束并未显著破坏策略的任务完成能力。结合 agent_collisions、pos_rew 以及训练曲线稳定性等指标可以进一步分析，reward shaping 对策略行为质量具有一定约束作用。

如果 APF-AW 显著低于 baseline，可以写成：

> APF-AW-MAPPO 引入更强的多势场自适应约束后，原始回报下降更明显。这表明在当前 VMAS/navigation 环境中，过强的安全与协同约束可能导致策略趋于保守，从而降低默认任务奖励。该结果从反面说明，低空多无人机路径规划中的奖励设计需要在任务完成、安全约束与轨迹质量之间进行平衡，而非简单叠加更多惩罚项。

## 6. 第四章建议结构

### 4.1 实验环境与参数设置

介绍 VMAS/navigation、BenchMARL、MAPPO、训练帧数、随机种子、网络结构等。

### 4.2 对比方法设置

介绍 baseline、fixed reward shaping、APF-AW-MAPPO 三组。

### 4.3 评价指标体系

重点强调本文采用多维指标，而不是单一 reward。

### 4.4 实验结果与曲线分析

展示 reward 曲线、最终均值、AUC、波动性。

### 4.5 多指标行为质量分析

展示 collisions、pos_rew、final_rew 等指标。

### 4.6 消融与讨论

讨论 fixed shaping 与 APF-AW 的差异，说明奖励约束强度过大可能导致保守策略。

### 4.7 本章小结

总结：baseline 在默认回报上占优，但 reward shaping 提供了行为约束视角；本文实验揭示了任务回报与安全/平滑约束之间的 trade-off。

## 7. 最终论文主张建议

更稳妥的主张是：

> 本文提出的 reward shaping 方法并非单纯追求更高的环境默认回报，而是尝试将低空无人机路径规划中更重要的目标推进、安全裕度与轨迹平滑性显式引入训练过程。实验表明，适度的 reward shaping 能够在基本保持任务完成能力的同时改变策略学习轨迹；而过强的多势场约束可能导致原始回报下降，说明多无人机协同路径规划中的奖励设计存在明显的多目标权衡关系。

这个说法比“本文方法全面优于 baseline”更安全、更真实，也更容易经得住答辩追问。
