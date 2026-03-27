## 项目目标
- 题目：基于 MAPPO 的城市低空场景多无人机协同避障与路径规划研究
- 代码路线：BenchMARL + VMAS
- 方法路线：先复现 MAPPO baseline，再加入 reward shaping
- 场景叙事：城市低空物流中的多无人机协同穿越障碍区域

## baseline
- MAPPO
- IPPO
- MADDPG

## 核心指标
- Success Rate
- Collision Rate
- Average Path Length
- Path Smoothness
- Episode Reward

## 本周计划
- 跑通 `vmas/balance` smoke test
- 切到 `vmas/navigation`
- 检查日志输出位置
- 整理实验记录模板
