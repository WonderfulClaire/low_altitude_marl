# Colab commands for APF-AW-MAPPO

下面不是 notebook 文件，而是一份可以直接复制到 Colab cell 里运行的命令清单。

## 1. Clone branch

```bash
!git clone -b feature/apf-aw-mappo https://github.com/WonderfulClaire/low_altitude_marl.git
%cd low_altitude_marl
```

如果仓库是 private，Colab 里需要用 GitHub token 或者先把 repo 临时设为 public。

## 2. Install dependencies

```bash
!pip install -r requirements.txt
```

## 3. Baseline

```bash
!python -m benchmarl.run \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

跑完后记录输出目录，例如：

```text
outputs/2026-xx-xx/xx-xx-xx
```

## 4. Fixed reward shaping v1

```bash
!python src/run_reward_shaping_v1.py \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

## 5. APF-AW-MAPPO full

```bash
!python src/run_apf_aw_mappo.py \
  --apf-total-frames 300000 \
  --apf-variant full \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

## 6. Optional ablations

### no_coop

```bash
!python src/run_apf_aw_mappo.py \
  --apf-total-frames 300000 \
  --apf-variant no_coop \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

### no_smooth

```bash
!python src/run_apf_aw_mappo.py \
  --apf-total-frames 300000 \
  --apf-variant no_smooth \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

### fixed multi-field

```bash
!python src/run_apf_aw_mappo.py \
  --apf-total-frames 300000 \
  --apf-variant fixed \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000 \
  seed=0
```

## 7. Compare runs

把下面路径替换成你实际的输出路径：

```bash
!python src/compare_apf_aw_runs.py \
  --run baseline=/content/low_altitude_marl/outputs/BASELINE_DIR \
  --run fixed=/content/low_altitude_marl/outputs/FIXED_DIR \
  --run apf_aw=/content/low_altitude_marl/outputs/APF_AW_DIR \
  --out /content/low_altitude_marl/results/apf_aw_compare
```

输出包括：

```text
results/apf_aw_compare/apf_aw_comparison_summary.csv
results/apf_aw_compare/pivot_final.csv
results/apf_aw_compare/pivot_best.csv
results/apf_aw_compare/pivot_last3_mean.csv
results/apf_aw_compare/pivot_auc.csv
results/apf_aw_compare/*.png
```

## 8. 建议优先跑法

时间紧的话，先跑三组：

```text
baseline
fixed reward shaping v1
APF-AW-MAPPO full
```

如果 APF-AW 的曲线明显更好，再补 no_coop/no_smooth 作为消融。
