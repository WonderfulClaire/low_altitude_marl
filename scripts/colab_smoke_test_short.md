# Colab smoke test (short runnable version)

这个版本用于在 Colab 上快速验证 BenchMARL + VMAS + MAPPO 训练链路是否打通。

保持当前路线：
- BenchMARL + VMAS
- MAPPO baseline
- 后续加入 reward shaping
- 不引入 3D dynamics

## 建议命令

```python
!git clone https://github.com/WonderfulClaire/low_altitude_marl.git
%cd low_altitude_marl
```

```python
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
```

```python
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
```

```python
import os
os.environ['WANDB_MODE'] = 'disabled'

!python -m benchmarl.run \
  algorithm=mappo \
  task=vmas/balance \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=60000
```
