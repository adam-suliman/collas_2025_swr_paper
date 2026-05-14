# ViT Incremental CIFAR Experiment Tracker

Generated from local result folders on 2026-05-14.

## Scope

This tracker includes only completed runs matching the canonical full incremental CIFAR setup:

- `num_epochs=2000`
- `initial_num_classes=5`
- `class_increase=5`
- `class_increase_frequency=100`
- `fixed_classes=false`
- `test_accuracy_per_epoch/index-*.npy` exists, has length 2000, and is nonzero for every epoch of the listed run index.

Excluded:

- Smoke runs.
- 500-epoch runs and 1000-epoch / 50-class-frequency runs.
- Truly unfinished runs with missing or incomplete metric arrays.
- Archived duplicate result trees, including `incremental_cifar_cbp_compact_archive`.

`Final test acc` is the final epoch test accuracy. For multi-seed entries, it is reported as mean +/- population standard deviation over the listed run indices.

## Baselines

| Setup | Key Params | Runs | Final Test Acc | Result Dir |
|---|---|---:|---:|---|
| Base | `reparam_ln=false`, `reset_network=false` | 0 | 0.4951 | [results/reparam_ln-False_reset_network-False](results/reparam_ln-False_reset_network-False) |
| Base | `reparam_ln=true`, `reset_network=false` | 0 | 0.5888 | [results/reparam_ln-True_reset_network-False](results/reparam_ln-True_reset_network-False) |
| Base | `reparam_ln=true`, `reset_network=false` | 0 | 0.5809 | [legacy final_runs/reparam_ln-True_reset_network-False](../../results/vit_incremental_cifar/final_runs/reparam_ln-True_reset_network-False) |
| Network reset | `reparam_ln=false`, `reset_network=true` | 0 | 0.5896 | [results/reparam_ln-False_reset_network-True](results/reparam_ln-False_reset_network-True) |
| Network reset | `reparam_ln=true`, `reset_network=true` | 1 | 0.5969 | [legacy final_runs/reparam_ln-True_reset_network-True](../../results/vit_incremental_cifar/final_runs/reparam_ln-True_reset_network-True) |
| Network reset | `reparam_ln=true`, `reset_network=true` | 0 | 0.5967 | [legacy final_runs/reparam_ln-True_reset_network-True_num_epochs-2000](../../results/vit_incremental_cifar/final_runs/reparam_ln-True_reset_network-True_num_epochs-2000) |

## Paper Baselines

| Setup | Key Params | Runs | Final Test Acc | Result Dir |
|---|---|---:|---:|---|
| SWR | `freq=128`, `factor=0.0005`, `utility=gradient`, `prune=threshold`, `strat=resample` | 0 | 0.5975 | [results/reparam_ln-True_reinit_freq-128_reinit_factor-0.0005_utility_function-gradient_pruning_method-threshold_reinit_strat-resample](results/reparam_ln-True_reinit_freq-128_reinit_factor-0.0005_utility_function-gradient_pruning_method-threshold_reinit_strat-resample) |
| Shrink and perturb | `parameter_noise_var=1e-08` | 0 | 0.5821 | [legacy final_runs/parameter_noise_var-1e-08](../../results/vit_incremental_cifar/final_runs/parameter_noise_var-1e-08) |

## CBP Sweep

| Setup | Key Params | Runs | Final Test Acc | Result Dir |
|---|---|---:|---:|---|
| CBP | `replacement_rate=1e-07`, `maturity=1000` | 0,1,2,3,4 | 0.5984 +/- 0.0038 | [parameter_sweeps/reparam_ln-True_replacement_rate-1e-07_maturity_threshold-1000](../../results/vit_incremental_cifar/parameter_sweeps/incremental_cifar/reparam_ln-True_replacement_rate-1e-07_maturity_threshold-1000) |
| CBP | `replacement_rate=1e-07`, `maturity=10000` | 0,1,2,3,4 | 0.5935 +/- 0.0019 | [parameter_sweeps/reparam_ln-True_replacement_rate-1e-07_maturity_threshold-10000](../../results/vit_incremental_cifar/parameter_sweeps/incremental_cifar/reparam_ln-True_replacement_rate-1e-07_maturity_threshold-10000) |
| CBP | `replacement_rate=1e-08`, `maturity=1000` | 0,1,2,3,4 | 0.5899 +/- 0.0040 | [parameter_sweeps/reparam_ln-True_replacement_rate-1e-08_maturity_threshold-1000](../../results/vit_incremental_cifar/parameter_sweeps/incremental_cifar/reparam_ln-True_replacement_rate-1e-08_maturity_threshold-1000) |
| CBP | `replacement_rate=1e-08`, `maturity=10000` | 0,1,2,3,4 | 0.4730 +/- 0.2315 | [parameter_sweeps/reparam_ln-True_replacement_rate-1e-08_maturity_threshold-10000](../../results/vit_incremental_cifar/parameter_sweeps/incremental_cifar/reparam_ln-True_replacement_rate-1e-08_maturity_threshold-10000) |
| CBP | `replacement_rate=5e-07`, `maturity=1000` | 0,1,2,3,4 | 0.5881 +/- 0.0017 | [parameter_sweeps/reparam_ln-True_replacement_rate-5e-07_maturity_threshold-1000](../../results/vit_incremental_cifar/parameter_sweeps/incremental_cifar/reparam_ln-True_replacement_rate-5e-07_maturity_threshold-1000) |
| CBP | `replacement_rate=5e-07`, `maturity=10000` | 0 | 0.5962 | [legacy final_runs/reparam_ln-True_replacement_rate-5e-07_maturity_threshold-10000](../../results/vit_incremental_cifar/final_runs/reparam_ln-True_replacement_rate-5e-07_maturity_threshold-10000) |
| CBP | `replacement_rate=5e-08`, `maturity=1000` | 0,1,2,3,4 | 0.5940 +/- 0.0020 | [parameter_sweeps/reparam_ln-True_replacement_rate-5e-08_maturity_threshold-1000](../../results/vit_incremental_cifar/parameter_sweeps/incremental_cifar/reparam_ln-True_replacement_rate-5e-08_maturity_threshold-1000) |
| CBP | `replacement_rate=5e-08`, `maturity=10000` | 0,1,2,3,4 | 0.5951 +/- 0.0026 | [parameter_sweeps/reparam_ln-True_replacement_rate-5e-08_maturity_threshold-10000](../../results/vit_incremental_cifar/parameter_sweeps/incremental_cifar/reparam_ln-True_replacement_rate-5e-08_maturity_threshold-10000) |

## RMT Baseline

| Setup | Key Params | Runs | Final Test Acc | Result Dir |
|---|---|---:|---:|---|
| Baseline | `n_mem=2`, `slow_freq=2`, `task_reset=true` | 0 | 0.5331 | [results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2](results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2) |
| Baseline | `n_mem=2`, `slow_freq=2`, `prior=true`, `task_reset=true` | 0 | 0.5398 | [results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2_rmt_use_learnable_memory_prior-True](results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2_rmt_use_learnable_memory_prior-True) |
| Baseline EMA | `n_mem=2`, `slow_freq=2`, `ema=0.0`, `task_reset=false` | 0 | 0.5504 | [results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2_rmt_baseline_memory_ema_beta-0.0](results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2_rmt_baseline_memory_ema_beta-0.0) |
| Baseline EMA | `n_mem=2`, `slow_freq=2`, `ema=0.9`, `task_reset=true` | 0 | 0.4670 | [results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2_rmt_baseline_memory_ema_beta-0.9](results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2_rmt_baseline_memory_ema_beta-0.9) |
| Baseline memory tokens | `n_mem=4`, `slow_freq=10`, `task_reset=true` | 0 | 0.5149 | [results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-10_rmt_n_mem-4](results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-10_rmt_n_mem-4) |
| Baseline memory tokens | `n_mem=8`, `slow_freq=2`, `task_reset=true` | 0 | 0.5071 | [results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2_rmt_n_mem-8](results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2_rmt_n_mem-8) |
| Baseline memory tokens | `n_mem=32`, `slow_freq=10`, `task_reset=true` | 0 | 0.5336 | [results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-10_rmt_n_mem-32](results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-10_rmt_n_mem-32) |

## RMT Fast Memory Core

| Setup | Key Params | Runs | Final Test Acc | Result Dir |
|---|---|---:|---:|---|
| Fast memory | `n_mem=2`, `fast_lr=0.1`, `slow_freq=2`, `prior=false`, `task_reset=true` | 0,1 | 0.6024 +/- 0.0000 | [results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_use_learnable_memory_prior-False](results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_use_learnable_memory_prior-False) |
| Fast memory | `n_mem=2`, `fast_lr=0.1`, `slow_freq=2`, `prior=false`, `task_reset=true`, `keep_all_checkpoints=true` | 0 | 0.6042 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_keep_all_experiment_checkpoints-True](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_keep_all_experiment_checkpoints-True) |
| Fast memory | `n_mem=2`, `fast_lr=0.1`, `slow_freq=2`, `prior=false`, `task_reset=true`, `fast_lr_min=0.1` | 2,3 | 0.5943 +/- 0.0040 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_min-0.1](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_min-0.1) |
| Fast memory | `n_mem=2`, `fast_lr=0.0`, `slow_freq=2`, `prior=false`, `task_reset=true` | 0 | 0.5923 | [results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.0_rmt_slow_update_freq-2_rmt_use_learnable_memory_prior-False](results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.0_rmt_slow_update_freq-2_rmt_use_learnable_memory_prior-False) |
| Fast memory | `n_mem=2`, `fast_lr=0.15`, `slow_freq=2`, `task_reset=true` | 0 | 0.5957 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.15_rmt_slow_update_freq-2](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.15_rmt_slow_update_freq-2) |
| Fast memory | `n_mem=2`, `fast_lr=0.1`, `slow_freq=1`, `prior=false`, `task_reset=true` | 0 | 0.5931 | [results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-1_rmt_use_learnable_memory_prior-False](results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-1_rmt_use_learnable_memory_prior-False) |
| Fast memory | `n_mem=2`, `fast_lr=0.1`, `slow_freq=5`, `prior=false`, `task_reset=true` | 0 | 0.6031 | [results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-5_rmt_use_learnable_memory_prior-False](results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-5_rmt_use_learnable_memory_prior-False) |
| Fast memory | `n_mem=2`, `fast_lr=0.1`, `slow_freq=10`, `prior=false`, `task_reset=true` | 0 | 0.5815 | [results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-10_rmt_use_learnable_memory_prior-False](results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-10_rmt_use_learnable_memory_prior-False) |
| Fast memory | `n_mem=2`, `fast_lr=0.15`, `slow_freq=10`, `task_reset=true` | 0 | 0.5559 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.15_rmt_slow_update_freq-10](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.15_rmt_slow_update_freq-10) |
| Fast memory prior | `n_mem=2`, `fast_lr=0.1`, `slow_freq=10`, `prior=true`, `task_reset=true` | 0 | 0.5509 | [results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-10_rmt_use_learnable_memory_prior-True](results/reparam_ln-True_model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-10_rmt_use_learnable_memory_prior-True) |
| Fast memory reset | `n_mem=2`, `fast_lr=0.1`, `slow_freq=2`, `prior=false`, `task_reset=false` | 0 | 0.5942 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_memory_reset_at_task_boundary-False](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_memory_reset_at_task_boundary-False) |

## RMT Fast Memory Schedules And Late Switches

| Setup | Key Params | Runs | Final Test Acc | Result Dir |
|---|---|---:|---:|---|
| Fast LR schedule | `constant 0.1`, switch task 15 to `fast_lr=0.05`, `slow_freq=4` | 0 | 0.5923 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.05](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.05) |
| Fast LR schedule | `constant 0.1`, switch task 15 to `fast_lr=0.08`, `slow_freq=4` | 0 | 0.5904 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.08](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.08) |
| Fast LR schedule | `constant 0.1`, switch task 15 to `fast_lr=0.12`, `slow_freq=4` | 0 | 0.5881 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.12](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.12) |
| Fast LR schedule | `constant 0.1`, switch task 15 to `fast_lr=0.15`, `slow_freq=4` | 0 | 0.6046 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.15](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.15) |
| Fast LR schedule | `constant 0.1`, switch task 15 to `fast_lr=0.2`, `slow_freq=4` | 0 | 0.5949 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.2](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.2) |
| Fast LR schedule | `constant 0.1`, switch task 15 to `fast_lr=0.5`, `slow_freq=4` | 0 | 0.5873 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.5](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-0.5) |
| Fast LR schedule | `constant 0.1`, switch task 15 to `fast_lr=1.0`, `slow_freq=4` | 0 | 0.5882 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-1.0](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-1.0) |
| Fast LR schedule | `constant 0.1`, switch task 15 to `fast_lr=5.0`, `slow_freq=4` | 0 | 0.5592 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-5.0](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-5.0) |
| Fast LR schedule | `constant 0.1`, switch task 15 to `fast_lr=10.0`, `slow_freq=4` | 0 | 0.5572 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-10.0](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-constant_rmt_fast_lr_switch_task-15_rmt_fast_lr_after_switch-10.0) |
| Task-local schedule | `task_cosine_decay`, `fast_lr=0.1`, `fast_lr_min=0.07`, `slow_freq=2` | 0 | 0.5984 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-task_cosine_decay_rmt_fast_lr_min-0.07](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-task_cosine_decay_rmt_fast_lr_min-0.07) |
| Task-local schedule | `task_linear_decay`, `fast_lr=0.1`, `fast_lr_min=0.07`, `slow_freq=2` | 0 | 0.5983 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-task_linear_decay_rmt_fast_lr_min-0.07](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_fast_lr_schedule-task_linear_decay_rmt_fast_lr_min-0.07) |
| Slow update switch | `slow_freq=2`, switch task 15 to `slow_freq=1` | 0 | 0.6055 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_slow_update_freq_switch_task-15_rmt_slow_update_freq_after_switch-1](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_slow_update_freq_switch_task-15_rmt_slow_update_freq_after_switch-1) |
| Slow update switch | `slow_freq=2`, switch task 15 to `slow_freq=4` | 0 | 0.5930 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_slow_update_freq_switch_task-15_rmt_slow_update_freq_after_switch-4](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_slow_update_freq_switch_task-15_rmt_slow_update_freq_after_switch-4) |
| Slow update switch | `slow_freq=2`, switch task 15 to `slow_freq=6` | 0 | 0.5853 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_slow_update_freq_switch_task-15_rmt_slow_update_freq_after_switch-6](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_slow_update_freq_switch_task-15_rmt_slow_update_freq_after_switch-6) |
| Slow update switch | `slow_freq=2`, switch task 15 to `slow_freq=10` | 0 | 0.5778 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_slow_update_freq_switch_task-15_rmt_slow_update_freq_after_switch-10](results/model_family-rmt_rmt_variant-fast_memory_rmt_fast_lr-0.1_rmt_slow_update_freq-2_rmt_slow_update_freq_switch_task-15_rmt_slow_update_freq_after_switch-10) |

## RMT Memory Token Ablations

| Setup | Key Params | Runs | Final Test Acc | Result Dir |
|---|---|---:|---:|---|
| Fast memory tokens | `n_mem=4`, `fast_lr=0.1`, `slow_freq=2`, `prior=false`, `task_reset=true` | 0 | 0.5934 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_n_mem-4_rmt_fast_lr-0.1_rmt_slow_update_freq-2](results/model_family-rmt_rmt_variant-fast_memory_rmt_n_mem-4_rmt_fast_lr-0.1_rmt_slow_update_freq-2) |
| Fast memory tokens | `n_mem=32`, `fast_lr=0.1`, `slow_freq=2`, `prior=false`, `task_reset=true` | 0 | 0.5657 | [results/model_family-rmt_rmt_variant-fast_memory_rmt_n_mem-32_rmt_fast_lr-0.1_rmt_slow_update_freq-2](results/model_family-rmt_rmt_variant-fast_memory_rmt_n_mem-32_rmt_fast_lr-0.1_rmt_slow_update_freq-2) |
| Baseline tokens | `n_mem=4`, `slow_freq=10`, `task_reset=true` | 0 | 0.5149 | [results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-10_rmt_n_mem-4](results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-10_rmt_n_mem-4) |
| Baseline tokens | `n_mem=8`, `slow_freq=2`, `task_reset=true` | 0 | 0.5071 | [results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2_rmt_n_mem-8](results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-2_rmt_n_mem-8) |
| Baseline tokens | `n_mem=32`, `slow_freq=10`, `task_reset=true` | 0 | 0.5336 | [results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-10_rmt_n_mem-32](results/model_family-rmt_rmt_variant-baseline_rmt_slow_update_freq-10_rmt_n_mem-32) |

## RMT Meta Fast Memory

| Setup | Key Params | Runs | Final Test Acc | Result Dir |
|---|---|---:|---:|---|
| Meta fast memory | `n_mem=2`, `fast_lr=0.1`, `slow_freq=1`, `prior=true`, `task_reset=true`, `write_steps=3`, `support_frac=0.5` | 0 | 0.5721 | [results/model_family-rmt_rmt_variant-meta_fast_memory_rmt_fast_lr-0.1_rmt_use_learnable_memory_prior-True_rmt_slow_update_freq-1_rmt_meta_write_steps-3_rmt_meta_support_fraction-0.5_num_epochs-2000](results/model_family-rmt_rmt_variant-meta_fast_memory_rmt_fast_lr-0.1_rmt_use_learnable_memory_prior-True_rmt_slow_update_freq-1_rmt_meta_write_steps-3_rmt_meta_support_fraction-0.5_num_epochs-2000) |
| Meta fast memory schedule | `n_mem=2`, `fast_lr=0.1`, `slow_freq=1`, `prior=true`, `task_reset=true`, `task_cosine_decay`, `fast_lr_min=0.07`, `write_steps=3`, `support_frac=0.5` | 0 | 0.5740 | [results/model_family-rmt_rmt_variant-meta_fast_memory_rmt_fast_lr-0.1_rmt_fast_lr_schedule-task_cosine_decay_rmt_fast_lr_min-0.07_rmt_slow_update_freq-1_rmt_meta_write_steps-3_rmt_meta_support_fraction-0.5](results/model_family-rmt_rmt_variant-meta_fast_memory_rmt_fast_lr-0.1_rmt_fast_lr_schedule-task_cosine_decay_rmt_fast_lr_min-0.07_rmt_slow_update_freq-1_rmt_meta_write_steps-3_rmt_meta_support_fraction-0.5) |

## Notes

- Full replay means final-task accuracy is not a strict continual-learning metric. Use task-start recovery, per-task AUC, and early-after-switch metrics when those summaries are available.
- Dynamic late-task switch runs are tracked as full ablations when their metric arrays are complete. They are not treated as incomplete solely because hyperparameters changed after task 15.
- Result directories under `results/vit_incremental_cifar/...` are legacy/original-paper locations. Result directories under `experiments/vit_incremental_cifar/results/...` are the newer RMT-oriented local outputs.
- The new `batch_recurrent` no-batch-average memory ablation has a config, but it is not listed here until a completed canonical run exists.
