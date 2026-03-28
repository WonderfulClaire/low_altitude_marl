# Colab navigation baseline (short runnable version)

这个版本用于在 Colab 上快速验证 `vmas/navigation` 任务下的 MAPPO baseline 是否能正常启动并完成短版训练。

保持当前路线：
- BenchMARL + VMAS
- MAPPO baseline
- 后续加入 reward shaping
- 不引入 3D dynamics

## 建议命令

```python
import os
os.environ['WANDB_MODE'] = 'disabled'

!python -m benchmarl.run \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=60000
```

## 当前已验证配置
- task: `vmas/navigation`
- algorithm: `mappo`
- render: `false`
- evaluation: `false`
- max_n_frames: `60000`
