# low_altitude_marl

毕业设计项目：**基于 MAPPO 的低空场景多无人机协同避障与路径规划研究**。

当前项目不再处于“只做 smoke test”的阶段，而是已经形成了一条比较清晰的主线：

- 使用 **VMAS** 作为多智能体仿真环境
- 选择 **`navigation`** 作为具体任务
- 使用 **BenchMARL** 作为训练 / benchmark 框架
- 以 **MAPPO** 作为 baseline
- 在 baseline 上加入 **reward shaping** 作为主要改进方向

---

## 1. 项目当前想解决什么问题？

本项目聚焦于：

> 在低空多无人机协同避障与路径规划任务中，如何基于多智能体强化学习训练出更安全、更高效、更平滑的导航策略。

在当前实现中，我们不直接把 VMAS/navigation 说成真实低空空域，而是把它视为：

> **低空多无人机协同避障与路径规划问题的二维抽象原型环境**。

它能够用较低成本验证策略学习方法在以下方面的表现：

- 任务完成能力
- 近障风险规避能力
- 轨迹平滑性
- 多 agent 共享空间下的协调能力

---

## 2. 为什么选择 VMAS/navigation？

VMAS 是一个向量化多智能体仿真环境库，包含多个任务场景。当前项目最终选择的是：

- **Environment**: VMAS
- **Task**: `navigation`

`navigation` 任务中，多个 agent 在带障碍的二维空间中各自导航到自己的目标点，同时需要避免碰撞并保持合理的运动行为。

这与本文的低空场景叙事可以对应为：

- obstacle → 建筑物 / 禁飞区 / 高风险区域
- goal → 配送点 / 巡检点 / 任务点
- collision → 空域冲突 / 飞行安全风险
- path length → 任务效率 / 能耗
- smoothness → 飞行稳定性与控制友好性

因此，`navigation` 是当前阶段最适合拿来做多无人机协同避障与路径规划抽象验证的任务。

---

## 3. 当前项目的两条核心实验线

当前项目已经明确为两条主线：

### 3.1 MAPPO baseline
原始 baseline 为：

- **MAPPO on VMAS/navigation**

它的作用是：
- 作为对照组
- 给出原始 reward 下的训练表现
- 为后续改进版提供比较基准

### 3.2 MAPPO + reward shaping
改进版在原始 navigation reward 的基础上，加入 reward shaping，用于增强：

- 过程反馈（解决稀疏奖励）
- 安全约束（增强近障惩罚）
- 轨迹质量（抑制抖动、鼓励平滑）

当前 reward shaping 终版构想以**简洁有效优先**为原则，核心包含 2–3 个 shaping term：

- `r_progress`：进度奖励
- `r_safety`：安全惩罚
- `r_smooth`：平滑惩罚

这条主线对应的核心问题是：

> 证明加入 reward shaping 后，相比原始 MAPPO baseline，策略能够学到更安全、更高效、更平滑的导航行为。

---

## 4. 当前 reward shaping 的设计思路

当前 reward shaping 的主要思路如下：

### 4.1 进度奖励 `r_progress`
```python
r_progress = d_prev - d_curr
```
其中 `d` 为 agent 到自身目标点的欧氏距离。

作用：
- 将“只有到终点才得到反馈”的稀疏奖励稠密化
- 每一步更接近目标都能获得正反馈
- 使 agent 更容易学会持续朝目标推进

### 4.2 安全惩罚 `r_safety`
推荐采用平滑形式：

```python
r_safety = -alpha * torch.relu(d_safe - d_obs)
```
其中 `d_obs` 为与最近障碍物或其他 agent 的距离。

作用：
- 不再只是“撞了才罚”
- 当 agent 进入危险区域时就提前给惩罚
- 更贴近低空飞行中的安全裕度约束

### 4.3 平滑惩罚 `r_smooth`
```python
r_smooth = -beta * torch.norm(action - prev_action)
```
作用：
- 抑制动作剧烈变化
- 减少轨迹抖动
- 让策略更符合无人机飞行平滑性要求

### 4.4 总体目标
最终希望改进版 reward 能够在原始 `r_task` 基础上，进一步强调：

- 导航过程中的持续推进
- 近障风险规避
- 飞行轨迹的稳定性与可解释性

---

## 5. 仓库当前结构说明

```text
low_altitude_marl/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── colab_smoke_test.ipynb
│   ├── colab_baseline_v1.ipynb
│   ├── colab_reward_shaping_v1.ipynb
│   └── colab_reward_shaping_run_v1.ipynb
└── src/
    ├── analyze_logs.py
    ├── metrics_summary.py
    ├── notes.md
    ├── reward_shaping_design.md
    ├── scenario_mapping.md
    └── run_reward_shaping_v1.py
```

### notebooks/
- `colab_smoke_test.ipynb`：早期 smoke test 与探索记录
- `colab_baseline_v1.ipynb`：干净的 baseline 运行 notebook
- `colab_reward_shaping_v1.ipynb`：改进版准备 notebook（设计与结构确认）
- `colab_reward_shaping_run_v1.ipynb`：改进版正式运行 notebook

### src/
- `metrics_summary.py`：读取并汇总 BenchMARL 输出结果
- `reward_shaping_design.md`：reward shaping 设计文档
- `scenario_mapping.md`：仿真任务与低空场景映射文档
- `run_reward_shaping_v1.py`：改进版 reward 的核心启动脚本

---

## 6. baseline 和改进版分别怎么跑？

### 6.1 baseline
使用 notebook：

- `notebooks/colab_baseline_v1.ipynb`

核心命令：

```python
!python -m benchmarl.run \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=60000 \
  seed=0
```

### 6.2 reward shaping 改进版
使用 notebook：

- `notebooks/colab_reward_shaping_run_v1.ipynb`

核心命令：

```python
!python src/run_reward_shaping_v1.py \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=60000 \
  seed=0
```

这里不是直接调用原始 `benchmarl.run`，而是先进入 `src/run_reward_shaping_v1.py`，由该脚本先 patch `VMAS/navigation` 的 reward，再在同一进程中启动训练。

---

## 7. 当前已经做完了什么？

截至当前，项目已经完成：

- baseline 主线确定：`MAPPO + VMAS/navigation`
- reward shaping 主线确定：`MAPPO + reward shaping`
- baseline notebook 单独整理完成
- 改进版 notebook 单独整理完成
- reward shaping 设计文档完成
- 场景映射文档完成
- 改进版 reward 启动脚本已经落成代码

也就是说，当前项目已经从“探索题目”进入“正式做实验和整理结果”的阶段。

---

## 8. 接下来最核心的工作

接下来最关键的不是继续扩题，而是形成完整实验闭环：

1. 跑完 **MAPPO baseline**
2. 跑完 **MAPPO + reward shaping**
3. 整理两者结果
4. 画出对比曲线
5. 补充至少一个效果展示：
   - success rate / collision rate / path length
   - 或轨迹可视化
6. 根据结果撰写实验分析与报告

---

## 9. 当前阶段最核心的一句话

当前项目的目标已经明确为：

> 证明在 VMAS/navigation 这一抽象多无人机协同导航任务中，基于 reward shaping 的 MAPPO 改进版相较于原始 MAPPO baseline，能够学习到更安全、更高效、更平滑的导航策略。

---

## 10. 备注

当前项目仍是毕业设计阶段的**仿真验证型研究**，不直接宣称已经完成真实无人机部署；但通过合理的场景映射、reward 设计和实验对比，希望尽可能把方法与低空无人机协同任务的现实需求联系起来。