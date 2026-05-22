# ViT Incremental CIFAR Experiment Tracker

Generated from local result folders on 2026-05-14. Paper-required seed coverage updated on 2026-05-22.

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

## Paper Required Seed Coverage

This table counts distinct `run_index` values for paper-relevant final comparisons. Duplicate directories with the same `run_index` are not counted as extra seeds. The target is at least 3 clean seeds per setup. Updated on 2026-05-22 from completed canonical 2000-epoch metric arrays.

| Role | Setup | Clean Runs | Final Test Acc | Missing For 3 | Status |
|---|---|---:|---:|---:|---|
| Core baseline | Base ViT + reparam-LN | 0,1,2 | 0.5891 +/- 0.0025 | 0 | Complete |
| Plasticity baseline | Base ViT + reparam-LN + full weight reset at task boundary | 0,1,2 | 0.5979 +/- 0.0016 | 0 | Complete |
| Original baseline | ReDO + reparam-LN | 0,1,2 | 0.5879 +/- 0.0040 | 0 | Complete if included |
| Original baseline | CBP best, `replacement_rate=1e-7`, `maturity=1000` | 0,1,2,3,4 | 0.5984 +/- 0.0038 | 0 | Complete |
| Original baseline | SWR paper setup | 0,1,2 | 0.5924 +/- 0.0038 | 0 | Complete |
| Optional baseline | Shrink and perturb | 0 | 0.5821 | 2 | Needs seeds 1,2 if included |
| RMT baseline | Baseline, `n_mem=2`, `slow_freq=2`, task reset | 1,2 | 0.5394 +/- 0.0077 | 1 | Needs clean seed 0, unless the old seed-0 metric is repaired/accepted despite overwritten config metadata |
| RMT baseline candidate | Baseline, `n_mem=2`, `slow_freq=2`, no task reset | 0 | 0.5504 | 2 | Needs seeds 1,2 if this becomes the baseline comparator |
| RMT final | Fast memory, `n_mem=2`, `fast_lr=0.1`, `slow_freq=2`, task reset | 0,1,2,3 | 0.5988 +/- 0.0054 | 0 | Complete; merges behavior-equivalent no-op config dirs |
| RMT ablation candidate | Fast memory, `n_mem=1`, `fast_lr=0.1`, `slow_freq=2`, task reset | 0 | 0.5988 | 2 | Needs seeds 1,2 if claiming one token is sufficient |
| RMT ablation candidate | Batch recurrent, `n_mem=2`, `slow_freq=2`, task reset | 0 | 0.4864 | 2 | Needs seeds 1,2 only if retained as a reported ablation |
| RMT optional | Meta fast memory | 0 | 0.5721 | 2 | Needs seeds 1,2 if included |

Current shortest path for a 3-seed main comparison using Base, CBP, SWR, RMT baseline task-reset, and RMT fast memory is:

- RMT baseline task-reset: run clean seed `0` if strict clean metadata is required. The old seed-0 metric exists, but its saved config was overwritten by a failed `n_mem=32` launch.
- If we repair/accept the old RMT baseline seed-0 metadata, the main comparison has enough seeds.

Additional ablation seeds:

- Fast memory `n_mem=1`: run seeds `1,2` if we want a supported one-token claim.
- Batch recurrent: run seeds `1,2` only if the poor seed-0 result is still worth reporting.
- Shrink-and-perturb: run seeds `1,2` if it remains in the paper comparison.
- Meta fast memory: run seeds `1,2` if included as an RMT variant.

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
- `batch_recurrent` now has completed seed 0 in the paper-required coverage table; it underperformed strongly, so only run more seeds if we decide to report it as an ablation.
- The paper-required seed coverage table is fresher than the older exhaustive inventory sections below it; prefer that table for current run planning.
