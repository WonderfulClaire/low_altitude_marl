# APF-AW-MAPPO 实验方案

## 1. 方法定位

当前主线不再是简单的固定权重 reward shaping，而是升级为：

> APF-AW-MAPPO：Adaptive Potential-Field Weighted MAPPO

中文可写为：

> 基于自适应势场权重的改进 MAPPO 算法

该方法参考多势场奖励塑形类 MAPPO 改进思路，但针对本文的 VMAS/navigation 抽象多无人机协同导航任务进行了轻量化适配。核心目标是解决固定 reward shaping 在训练早期可能压制探索、导致最终回报不稳定的问题。

## 2. 核心奖励设计

原始 MAPPO 使用环境默认奖励：

```text
r_task
```

APF-AW-MAPPO 在此基础上加入四类势场项：

```text
r_total = r_task
        + w_p(t) * r_progress
        - w_s(t) * r_safety
        - w_m(t) * r_smooth
        - w_c(t) * r_coop
```

各项含义如下：

| 奖励项 | 含义 | 作用 |
|---|---|---|
| `r_progress` | 目标接近势场 | 鼓励无人机持续靠近目标点 |
| `r_safety` | 静态障碍安全势场 | 鼓励无人机远离障碍物/禁飞区 |
| `r_smooth` | 动作平滑势场 | 抑制动作剧烈变化，提高轨迹平滑性 |
| `r_coop` | 多机协同间距势场 | 减少无人机之间过近造成的空域冲突 |

## 3. 自适应权重机制

固定 reward shaping 的问题是：训练一开始 agent 还不会到达目标，如果同时施加强安全惩罚和平滑惩罚，会让探索过程变得更加困难。

因此 APF-AW-MAPPO 使用阶段式权重：

```text
训练早期：提高目标接近权重，降低安全/平滑/协同惩罚
训练中期：逐步提高安全和协同权重
训练后期：进一步加入平滑约束，使轨迹质量更高
```

对应到论文表述，可以写成：

> 本文采用课程式自适应权重调度机制，在训练初期优先缓解稀疏奖励带来的探索困难，在策略具备基本到达能力后逐步强化安全裕度、轨迹平滑性与多机协同约束。

## 4. 推荐实验组

建议正式实验至少跑以下三组：

| 组别 | 命令入口 | 方法含义 |
|---|---|---|
| A | `python -m benchmarl.run ...` | 原始 MAPPO baseline |
| B | `python src/run_reward_shaping_v1.py ...` | 固定权重 reward shaping |
| C | `python src/run_apf_aw_mappo.py --apf-variant full ...` | APF-AW-MAPPO 主方法 |

如果时间允许，补充两组消融：

| 组别 | 命令入口 | 方法含义 |
|---|---|---|
| D | `--apf-variant no_coop` | 去掉多机协同势场 |
| E | `--apf-variant no_smooth` | 去掉动作平滑势场 |
| F | `--apf-variant fixed` | 多势场固定权重版本 |

## 5. 推荐训练命令

### 5.1 MAPPO baseline

```bash
python -m benchmarl.run \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

### 5.2 固定 reward shaping 旧版本

```bash
python src/run_reward_shaping_v1.py \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

### 5.3 APF-AW-MAPPO 主方法

```bash
python src/run_apf_aw_mappo.py \
  --apf-total-frames 300000 \
  --apf-variant full \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

### 5.4 消融：去掉协同势场

```bash
python src/run_apf_aw_mappo.py \
  --apf-total-frames 300000 \
  --apf-variant no_coop \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

### 5.5 消融：去掉平滑势场

```bash
python src/run_apf_aw_mappo.py \
  --apf-total-frames 300000 \
  --apf-variant no_smooth \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

## 6. 最终论文中建议报告的指标

不要只报告 `episode_reward_mean`。建议第四章至少使用以下指标：

| 指标 | 说明 | 论文意义 |
|---|---|---|
| Episode Reward Mean | 平均回报 | 反映训练目标优化效果 |
| Best Reward | 最优回报 | 反映策略峰值性能 |
| Last-3 Mean | 末期稳定均值 | 避免只看单点最终值 |
| AUC | 训练曲线面积 | 同时反映收敛速度与总体表现 |
| Volatility | 曲线波动度 | 反映训练稳定性 |
| Collision/Distance/Smoothness | 若日志可得 | 支撑路径规划质量分析 |

## 7. 第四章可写的实验故事线

建议按照以下逻辑组织：

```text
1. 标准 MAPPO 能够完成 VMAS/navigation 任务，但主要优化环境默认回报。
2. 固定 reward shaping 虽然引入了目标推进、安全和平滑约束，但在训练早期可能压制探索，因此最终回报不一定优于 baseline。
3. APF-AW-MAPPO 通过自适应权重调度，在训练早期优先解决目标到达问题，在中后期逐步强化安全、平滑和多机协同约束。
4. 实验从最终回报、收敛速度、曲线稳定性和路径质量指标多角度验证方法有效性。
```

## 8. 本科毕设创新点表述

建议写成：

> 本文并未简单地将多个奖励项以固定权重叠加，而是根据多无人机协同导航任务的训练阶段差异，提出一种自适应势场权重调度机制。该机制在训练初期强调目标接近奖励，以缓解稀疏奖励下的探索困难；在训练中后期逐步提高安全、平滑与协同约束权重，使策略在完成导航任务的同时更符合低空飞行对安全裕度和控制平稳性的要求。
