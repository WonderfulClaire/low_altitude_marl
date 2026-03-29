# Scenario Mapping

## Purpose
将当前使用的 VMAS/navigation 多智能体导航任务映射为低空多无人机协同避障与路径规划的抽象原型场景，用于后续论文写作、答辩叙事和 reward shaping 设计。

## Mapping
### Environment elements
- Obstacles:
  interpreted as buildings, no-fly zones, risky airspace blocks
- Goals:
  interpreted as delivery targets, inspection points, mission destinations
- Agents:
  interpreted as multiple UAVs sharing the same low-altitude airspace

### Safety-related interpretation
- Collision:
  interpreted as airspace conflict or flight safety incident
- Near-obstacle region:
  interpreted as risky corridor / building boundary / danger margin

### Efficiency-related interpretation
- Path length:
  interpreted as mission efficiency and energy cost
- Time to goal:
  interpreted as task completion efficiency

### Trajectory-related interpretation
- Smoothness:
  interpreted as flight stability and controllability
- Large action change:
  interpreted as abrupt maneuvering or unstable control

## Narrative use in thesis
后续在论文和汇报中，不直接将 VMAS/navigation 表述为真实低空环境，而表述为：

> VMAS/navigation 是低空多无人机协同避障与路径规划问题的二维抽象原型环境，用于在可控条件下验证策略学习方法在安全性、效率与轨迹质量方面的表现。

## Why this mapping matters
1. 避免论文叙事停留在纯 toy simulator 层面。
2. 为 reward shaping 设计提供现实语义。
3. 为真实数据集与仿真任务之间建立解释桥梁。
4. 为答辩中的“落地价值”问题提供统一口径。
