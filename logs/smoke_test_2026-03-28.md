# Smoke test log — 2026-03-28

## Goal
在 Colab 上验证 BenchMARL + VMAS + MAPPO 的基础训练链路是否打通。

## Initial issue
默认命令在 Colab 上会在 evaluation/render 阶段报错：
- VMAS 渲染依赖 OpenGL / GLU
- Colab 无头环境缺少 `GLU`
- 典型报错：`ImportError: Library "GLU" not found.`

## Fix
将 smoke test 配置改为：
- `experiment.render=false`
- `experiment.evaluation=false`
- `experiment.max_n_frames=60000`
- `WANDB_MODE=disabled`

## Effective command
```bash
python -m benchmarl.run \
  algorithm=mappo \
  task=vmas/balance \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=60000
```

## Result
- 训练主循环正常启动
- smoke test 成功跑通
- 说明 GitHub 仓库、Colab 环境、BenchMARL、VMAS、MAPPO 链路已基本打通

## Next step
- 切换到 `vmas/navigation`
- 保持 `render=false`、`evaluation=false`
- 跑正式 baseline 的短版验证
