# No more runs: next steps for Chapter 4

## Conclusion

No additional training runs are needed for the thesis at this stage.

Already completed:

1. MAPPO baseline from the original thesis experiment.
2. MAPPO + fixed reward shaping from the original thesis experiment.
3. APF-AW exploratory run, which should not be used as the main thesis result.
4. Re-run MAPPO baseline, confirming the baseline result is reproducible at approximately 0.963.

## What to do next

The next work is not running more experiments, but converting existing runs into Chapter 4 materials:

1. Extract final scalar results from the existing baseline and fixed reward-shaping logs.
2. Build a multi-metric evaluation table.
3. Add a new evaluation-metric subsection in Chapter 4.
4. Add a short baseline reproducibility paragraph.
5. Add an explanation of why reward shaping does not exceed baseline in raw reward.
6. Add a limitation/future-work paragraph about fixed reward weights.

## Experiments to exclude from the main thesis

APF-AW should not be included in the main text because it introduces a new branch of method design and its result is significantly lower than the fixed reward-shaping version. It may be mentioned only as an internal exploration or not mentioned at all.

## Final thesis line

The thesis should focus on baseline vs fixed reward shaping, and present the contribution as a reproducible reward-design and evaluation analysis rather than claiming that reward shaping outperforms baseline in raw episode reward.
