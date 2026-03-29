# Reward Shaping v1

## Objective
在 VMAS/navigation 的 MAPPO baseline 上，引入更贴近低空多无人机协同飞行任务的奖励设计，以提升策略在安全性、效率与轨迹质量方面的表现。

## Reward terms
### 1. Goal reward
- 无人机成功到达目标点时给予较大正奖励。
- 用于突出任务完成的重要性。

### 2. Progress reward
- 若当前时间步相比上一步更接近目标点，则给予小幅正奖励。
- 用于缓解稀疏奖励问题，加速训练收敛。

### 3. Collision penalty
- 与障碍物或其他 agent 发生碰撞时给予较大负奖励。
- 用于强调飞行安全约束。

### 4. Risk penalty
- 当无人机距离障碍物过近时给予额外惩罚。
- 用于模拟低空场景中的高风险区域、建筑边界和禁飞区邻近效应。

### 5. Smoothness penalty
- 当连续两个时间步动作变化过大时给予惩罚。
- 用于鼓励更平滑的轨迹，减少抖动与急转弯。

## Proposed formulation
可以将总奖励写成：

R = w1 * R_goal + w2 * R_progress - w3 * R_collision - w4 * R_risk - w5 * R_smooth

其中各权重后续通过实验进行调节。

## Low-altitude scenario mapping
- obstacle -> 建筑物 / 禁飞区 / 高风险区域
- goal -> 配送点 / 巡检点 / 任务点
- collision -> 空域冲突 / 飞行安全风险
- path smoothness -> 飞行稳定性与控制友好性
- progress -> 任务推进效率

## Current plan
1. 先保留原始 MAPPO baseline 作为对照组。
2. 在不推翻现有训练平台的前提下，逐步将 reward shaping 引入任务。
3. 后续通过 episode_reward_mean、reward_mean、collision 相关指标和轨迹可视化比较改进前后的效果。
