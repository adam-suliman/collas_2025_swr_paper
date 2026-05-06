# RMT-FastMemory PMNIST Runbook

## 1) What Was Added
- `src/networks/permuted_mnist_rmt_network.py`
  - `MemoryTransformer` with patch embedding, memory tokens, Transformer encoder, classifier head.
- `experiments/permuted_mnist/permuted_mnist_experiment.py`
  - `model_family` switch: `mlp` (legacy) vs `rmt` (new).
  - `rmt_variant` switch: `baseline` vs `fast_memory`.
  - GPU selection via existing `--gpu_index`.
  - New RMT summaries:
    - `rmt_memory_norm_per_checkpoint`
    - `rmt_memory_update_norm_per_checkpoint`
- `src/networks/__init__.py`
  - exports `MemoryTransformer`.

## 2) New Config Fields
- `model_family`: `mlp` or `rmt` (default is `mlp`)
- `rmt_variant`: `baseline` or `fast_memory`
- `rmt_patch_size` (default `4`)
- `rmt_d_model` (default `64`)
- `rmt_n_layers` (default `2`)
- `rmt_n_heads` (default `4`)
- `rmt_mlp_ratio` (default `2.0`)
- `rmt_n_mem` (default `2`)
- `rmt_fast_lr` (default `0.1`)
- `rmt_slow_update_freq` (default `10`)
- `rmt_memory_reset_at_task_boundary` (default `true`)
- `rmt_clip_memory_grad` (default `null`)

## 3) Algorithm-to-Code Mapping
- Baseline (Variant A):
  - forward with memory tokens and request encoded memory
  - update memory by encoder-carry:
    - `m_{t+1} = mean_batch(encoded_memory_tokens_t)`
  - no gradient update on memory.
- FastMemory (Variant B):
  - memory has `requires_grad=True`
  - after backward:
    - `m_{t+1} = m_t - rmt_fast_lr * grad_m`
  - optional clipping through `rmt_clip_memory_grad`
  - memory is detached/re-enabled each step.
- Slow update:
  - slow weights (`theta`) use existing optimizer settings
  - update every `rmt_slow_update_freq` steps.

## 4) Provided Configs
Located at:
- `experiments/permuted_mnist/config/local/single_runs/large_network/rmt_fast_memory.json`
- `experiments/permuted_mnist/config/local/single_runs/large_network/rmt_baseline.json`
- `experiments/permuted_mnist/config/local/single_runs/large_network/rmt_fast_memory_smoke.json`
- `experiments/permuted_mnist/config/local/single_runs/large_network/rmt_baseline_smoke.json`

## 5) Linux Server Setup
```bash
git clone <your-repo-url>
cd collas_2025_swr_paper
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 6) Run Commands
Single run:
```bash
python experiments/permuted_mnist/permuted_mnist_experiment.py \
  -c experiments/permuted_mnist/config/local/single_runs/large_network/rmt_fast_memory.json \
  -i 0 --gpu_index 0 -v
```

Baseline run:
```bash
python experiments/permuted_mnist/permuted_mnist_experiment.py \
  -c experiments/permuted_mnist/config/local/single_runs/large_network/rmt_baseline.json \
  -i 0 --gpu_index 0 -v
```

3-GPU parallel seeds:
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/permuted_mnist/permuted_mnist_experiment.py -c experiments/permuted_mnist/config/local/single_runs/large_network/rmt_fast_memory.json -i 0 --gpu_index 0 > run_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python experiments/permuted_mnist/permuted_mnist_experiment.py -c experiments/permuted_mnist/config/local/single_runs/large_network/rmt_fast_memory.json -i 1 --gpu_index 0 > run_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python experiments/permuted_mnist/permuted_mnist_experiment.py -c experiments/permuted_mnist/config/local/single_runs/large_network/rmt_fast_memory.json -i 2 --gpu_index 0 > run_gpu2.log 2>&1 &
wait
```

## 7) Results and Analysis
- Results are stored under default PMNIST `results` path (or custom `results_path` in config).
- New summary arrays are saved the same way as existing summaries.

Example analysis config snippet for new RMT summaries:
```json
{
  "results_dir": "experiments/permuted_mnist/results/<your_run_dir>",
  "parameter_combinations": ["<your_parameter_combination_dir>"],
  "summary_names": [
    "train_accuracy_per_checkpoint",
    "rmt_memory_norm_per_checkpoint",
    "rmt_memory_update_norm_per_checkpoint"
  ],
  "plot_parameters": {
    "xlabel": "Checkpoint",
    "ylabel": "Value"
  }
}
```

## 8) Manual Transfer Checklist
Copy these files to your Linux machine:
- Updated:
  - `experiments/permuted_mnist/permuted_mnist_experiment.py`
  - `src/networks/__init__.py`
- New:
  - `src/networks/permuted_mnist_rmt_network.py`
  - `experiments/permuted_mnist/config/local/single_runs/large_network/rmt_fast_memory.json`
  - `experiments/permuted_mnist/config/local/single_runs/large_network/rmt_baseline.json`
  - `experiments/permuted_mnist/config/local/single_runs/large_network/rmt_fast_memory_smoke.json`
  - `experiments/permuted_mnist/config/local/single_runs/large_network/rmt_baseline_smoke.json`
  - `experiments/permuted_mnist/RMT_RUNBOOK.md`

Syntax check:
```bash
python -m py_compile \
  experiments/permuted_mnist/permuted_mnist_experiment.py \
  src/networks/permuted_mnist_rmt_network.py
```
