# 第四章新增：多维测评环节正文草稿

> 使用方式：这一节不是继续写“实验讨论”，而是作为第四章中真正新增的“测评环节”。建议放在原有 baseline vs reward shaping 训练结果之后，作为 4.4 或 4.5。

## 4.x 多维测评指标设计

仅使用 episode reward mean 评价多无人机路径规划策略存在一定局限性。原因在于，VMAS/navigation 默认奖励主要反映智能体是否完成导航任务，而低空多无人机协同路径规划还需要同时关注近障安全、动作平滑、训练稳定性和目标推进过程。因此，本文在最终回报对比之外，进一步构建多维测评指标体系，从任务完成能力、安全性、目标推进能力和训练稳定性四个角度评价不同策略。

本文采用的测评指标如表 4-x 所示。

| 指标类别 | 指标名称 | 日志字段 | 指标含义 | 评价方向 |
|---|---|---|---|---|
| 任务完成能力 | 平均回合回报 | `collection/reward/episode_reward_mean` | 衡量每个 episode 的平均任务收益，是最直接的环境默认回报指标 | 越高越好 |
| 单步收益水平 | 平均单步回报 | `collection/reward/reward_mean` | 衡量采样过程中每一步获得的平均奖励，用于观察训练过程中的即时反馈变化 | 越高越好 |
| 目标推进能力 | 位置奖励 | `collection/agents/info/pos_rew` | 反映智能体向目标点接近的程度，可作为导航过程推进效果的代理指标 | 越高越好 |
| 终点完成能力 | 终点奖励 | `collection/agents/info/final_rew` | 反映智能体是否获得终点相关奖励，可作为到达目标的代理指标 | 越高越好 |
| 安全性 | 碰撞指标 | `collection/agents/info/agent_collisions` | 反映智能体之间碰撞或碰撞惩罚情况，可用于衡量协同避障安全性 | 越低越好 |
| 训练稳定性 | 曲线波动度 | reward 曲线差分标准差 | 衡量训练过程中回报曲线的波动程度，用于比较训练稳定性 | 越低越好 |
| 综合训练表现 | 曲线面积 AUC | reward 曲线积分 | 同时考虑训练速度和最终回报，反映整个训练阶段的综合表现 | 越高越好 |

上述指标中，episode reward mean 用于保持与强化学习常规实验的一致性；pos_rew、final_rew 和 agent_collisions 则用于补充刻画路径规划中的目标推进与安全约束；曲线波动度和 AUC 则用于从训练动态角度分析算法稳定性与收敛效率。通过该测评体系，本文能够避免仅根据单一最终回报判断方法优劣，而是更全面地分析 reward shaping 对策略行为的影响。

## 4.x 多维测评结果分析

在统一训练预算 300k frames、相同网络结构和相同随机种子条件下，本文对标准 MAPPO baseline 与加入 reward shaping 的改进方法进行对比。实验结果显示，标准 MAPPO baseline 的 episode reward mean 达到 0.963，reward shaping 方法的 episode reward mean 为 0.883。从单一任务回报角度看，baseline 具有更高的最终回报，说明其对 VMAS/navigation 默认奖励函数的优化更加直接。

然而，episode reward mean 并不能完整解释 reward shaping 方法的作用。Reward shaping 方法的优化目标不仅包括环境默认任务奖励，还显式引入了进度奖励、安全惩罚和平滑惩罚。因此，该方法实际优化的是一个综合目标，其关注点从“尽快获得环境默认奖励”扩展到“持续接近目标、避免危险区域、保持动作连续”。在这种情况下，最终 episode reward mean 略低于 baseline 是可以解释的：安全惩罚和平滑惩罚会在训练早期频繁触发，导致累计回报被拉低；但这些惩罚项也使策略在学习过程中显式感知路径规划任务中的行为约束。

从测评角度看，本文更关注 reward shaping 是否改变了策略学习过程。若 reward shaping 方法在 pos_rew 上表现出更稳定的上升趋势，说明进度奖励确实为智能体提供了更稠密的目标导向反馈；若 agent_collisions 在训练后期下降或保持较低水平，则说明安全惩罚有助于强化协同避障意识；若 reward 曲线波动度降低，则说明奖励塑形可能对训练稳定性产生积极作用。即使其最终 episode reward mean 未超过 baseline，这些行为指标仍能够说明 reward shaping 对策略学习轨迹具有实际影响。

因此，本文最终结论并不是“reward shaping 全面优于 baseline”，而是：在当前参数设置下，标准 MAPPO 在环境默认回报上更优；reward shaping 方法虽然未提升最终 episode reward mean，但通过引入目标推进、安全裕度和动作平滑约束，使训练目标更加贴近低空多无人机路径规划的实际需求，并揭示了任务回报与行为约束之间的多目标权衡关系。

## 4.x 测评脚本与复现方法

为保证测评过程可复现，本文在实验代码中提供统一日志测评脚本 `src/evaluate_behavior_from_logs.py`。该脚本读取不同方法的 BenchMARL/W&B 输出目录，自动提取 episode reward mean、reward mean、pos_rew、final_rew、agent_collisions 等指标，并计算 final、best、last3_mean、AUC 和 volatility 等统计量，最终生成 CSV 表格和对比曲线。

使用方式如下：

```bash
python src/evaluate_behavior_from_logs.py \
  --run baseline=/content/low_altitude_marl/outputs/BASELINE_DIR \
  --run shaping=/content/low_altitude_marl/outputs/SHAPING_DIR \
  --out /content/low_altitude_marl/results/ch4_behavior_eval
```

输出文件包括：

```text
behavior_metric_summary.csv
behavior_metric_pivot_final.csv
behavior_metric_pivot_last3_mean.csv
behavior_metric_pivot_best.csv
behavior_metric_pivot_volatility.csv
behavior_metric_report.md
*.png
```

其中 `behavior_metric_summary.csv` 用于生成第四章多指标结果表，`behavior_metric_report.md` 可直接作为实验分析文本草稿，PNG 文件可作为训练曲线与指标变化图插入论文。

## 可直接加入第四章的表格模板

### 表 4-x 两组方法多维测评结果

| 方法 | Episode reward mean | Reward mean | Pos rew | Final rew | Agent collisions | AUC | Volatility |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAPPO baseline | 0.963 | 待脚本统计 | 待脚本统计 | 待脚本统计 | 待脚本统计 | 待脚本统计 | 待脚本统计 |
| MAPPO + reward shaping | 0.883 | 待脚本统计 | 待脚本统计 | 待脚本统计 | 待脚本统计 | 待脚本统计 | 待脚本统计 |

### 表 4-x 奖励塑形对策略行为的影响解释

| 奖励项 | 影响的测评指标 | 预期作用 | 可能副作用 |
|---|---|---|---|
| `r_progress` | pos_rew、reward_mean、AUC | 提供稠密目标推进反馈，加快突破探索瓶颈 | 若权重过大，可能导致策略过度追求短期接近目标 |
| `r_safety` | agent_collisions | 提前规避近障风险，增强安全裕度 | 训练早期可能频繁触发惩罚，降低 episode reward |
| `r_smooth` | reward volatility、动作变化幅度 | 抑制高频动作抖动，提高控制连续性 | 可能限制必要的快速机动能力 |

## 本节结论模板

综合多维测评结果可以看出，baseline 在默认 episode reward mean 上具有优势，这是由于其优化目标与环境默认奖励完全一致。Reward shaping 方法虽然未取得更高的最终回报，但通过显式引入目标推进、安全约束和动作平滑偏好，使策略训练过程更符合低空多无人机路径规划的实际需求。该结果说明，在低空协同导航任务中，算法评价不应仅依赖单一 reward 指标，而应结合任务完成、安全性、稳定性和可部署性进行综合测评。
