# EVRPTW Cus100 Ablation Launch Scripts

These scripts launch the EVRPTW Cus100 ablation batch from
`/data/Maojie/experiment_spec.md`. They are designed for multi-server runs where
each physical server executes one script and distributes jobs across its local
GPUs in waves.

The scripts run on the `ablation` branch of CaliRoute. The expected default data
layout is:

```bash
/data/Maojie/CaliRoute
/data/Maojie/Routing-D
```

If the dataset is elsewhere, set `DATA_ROOT=/path/to/Routing-D`.

## Quick Start

Run from the repository root:

```bash
cd /data/Maojie/CaliRoute
git checkout ablation
git pull
```

Check the script interface:

```bash
bash scripts/ablation/server2_rdi_4gpu.sh help
```

Each script accepts one stage argument and detaches under `nohup` by default:

```bash
bash scripts/ablation/serverX_*.sh init
bash scripts/ablation/serverX_*.sh ppo
bash scripts/ablation/serverX_*.sh offline
bash scripts/ablation/serverX_*.sh all
```

Use `--foreground` or `--no-detach` when you want to debug in the current terminal:

```bash
bash scripts/ablation/serverX_*.sh ppo --foreground
```

- `init`: train the shared PPO epoch-100 initial checkpoint only.
- `ppo`: ensure the shared PPO initial checkpoint, then run this server's PPO
  ablation jobs.
- `offline`: wait for the shared PPO initial checkpoint, then run non-PPO jobs
  such as SL-PPO, DAPG, or AWBC.
- `all`: run init if needed, then run both PPO and offline jobs for that server.

Use the two-stage flow for any server whose ablation group contains offline
methods:

```bash
bash scripts/ablation/serverX_*.sh ppo
bash scripts/ablation/serverX_*.sh offline
```

The offline stage uses `--init-checkpoint` from the PPO epoch-100 checkpoint.
It does not retrain PPO.

## Server Assignment

| Physical server | Script | GPUs | New jobs | Spec group |
| --- | --- | --- | ---: | --- |
| Server1 | `server1_expert_budget_incumbent_4gpu.sh` | `0,1,2,3` | 10 | expert budget x incumbent robustness |
| Server2 | `server2_rdi_4gpu.sh` | `0,1,2,3` | 8 | RDI ablation |
| Server3 | `server3_agda_progressive_slppo_4gpu.sh` | `0,1,2,3` | 9 | AGDA + progressive + SL-PPO advantage |
| Server4 | `server4_ntraj_feature_groups_3gpu.sh` | `0,1,2` | 6 | n_traj + AGDA feature groups |

Stage split:

| Server | `ppo` stage | `offline` stage |
| --- | --- | --- |
| Server1 | shared PPO init only | 10 expert-budget x incumbent SL-PPO jobs |
| Server2 | 8 RDI PPO jobs from scratch | no non-PPO jobs |
| Server3 | 4 AGDA/progressive PPO jobs | 5 progressive/advantage SL-PPO jobs |
| Server4 | 3 AGDA feature-group PPO jobs | 3 n_traj SL-PPO jobs |

## Run Templates

### Server1: Expert Budget x Incumbent

Phase 1 creates or verifies the shared PPO initial checkpoint:

```bash
cd /data/Maojie/CaliRoute
bash scripts/ablation/server1_expert_budget_incumbent_4gpu.sh ppo
```

Phase 2 launches the 10 SL-PPO offline jobs:

```bash
cd /data/Maojie/CaliRoute
bash scripts/ablation/server1_expert_budget_incumbent_4gpu.sh offline
```

Jobs:

| Tag template | Method | Main knobs |
| --- | --- | --- |
| `s1_budget60_incon` | SL-PPO | `checkpoint_s=60`, incumbent on |
| `s1_budget60_incoff` | SL-PPO | `checkpoint_s=60`, incumbent off |
| `s1_budget300_incon` | SL-PPO | `checkpoint_s=300`, incumbent on |
| `s1_budget300_incoff` | SL-PPO | `checkpoint_s=300`, incumbent off |
| `s1_budget900_incon` | SL-PPO | `checkpoint_s=900`, incumbent on |
| `s1_budget900_incoff` | SL-PPO | `checkpoint_s=900`, incumbent off |
| `s1_budget3600_incon` | SL-PPO | `checkpoint_s=3600`, incumbent on |
| `s1_budget3600_incoff` | SL-PPO | `checkpoint_s=3600`, incumbent off |
| `s1_budget7200_incon` | SL-PPO | `checkpoint_s=7200`, incumbent on |
| `s1_budget7200_incoff` | SL-PPO | `checkpoint_s=7200`, incumbent off |

To change the budgets:

```bash
BUDGETS="60 300 900" bash scripts/ablation/server1_expert_budget_incumbent_4gpu.sh offline
```

### Server2: RDI

Server2 is PPO-only, so `offline` is intentionally a no-op. By default it
launches the full Section 5.3 RDI table: 8 EVRPTW Cus100, single-seed, plain PPO
runs with AGDA disabled. These RDI rows train from scratch and do not use the
shared PPO init checkpoint.

```bash
cd /data/Maojie/CaliRoute
bash scripts/ablation/server2_rdi_4gpu.sh 0,1,2,3 all
```

The preferred interface is positional:

```bash
bash scripts/ablation/server2_rdi_4gpu.sh GPU_ID RDI_OPTION RDI_EMBEDDING RDI_ENCODER_SINKHORN RDI_ENCODER_BIAS
```

- `GPU_ID`: one GPU id such as `2`, or a comma list such as `0,1,2,3`.
- `RDI_OPTION`: `all`, `base`, `euclidean`, or `graph`.
- `RDI_EMBEDDING`: `true` enables SVD embedding; `false` disables it.
- For `graph`, the three switches are `true|false`; at least one must be `true`.
- For `base` and `euclidean`, the switch arguments can be omitted or left `false`.

`RDI_OPTION=all` expands to these rows:

| Tag | Distance option | Embedding SVD | Encoder Sinkhorn | Encoder bias | Method |
| --- | --- | --- | --- | --- | --- |
| `s2_base` | base | false | false | false | PPO |
| `s2_euclidean` | euclidean | false | false | true | PPO |
| `s2_embedding_svd_only` | graph | true | false | false | PPO |
| `s2_encoder_sinkhorn_only` | graph | false | true | false | PPO |
| `s2_encoder_bias_only` | graph | false | false | true | PPO |
| `s2_embedding_svd_encoder_sinkhorn` | graph | true | true | false | PPO |
| `s2_embedding_svd_encoder_bias` | graph | true | false | true | PPO |
| `s2_embedding_svd_encoder_sinkhorn_encoder_bias` | graph | true | true | true | PPO |

To run one row:

```bash
bash scripts/ablation/server2_rdi_4gpu.sh 0 base false false false
bash scripts/ablation/server2_rdi_4gpu.sh 1 euclidean false false false
bash scripts/ablation/server2_rdi_4gpu.sh 2 graph true false true
```

The old stage/env interface still works for compatibility, for example
`RDI_OPTION=graph RDI_EMBEDDING_SVD=true RDI_ENCODER_BIAS=true bash scripts/ablation/server2_rdi_4gpu.sh ppo`.

This script also writes one nearest-neighbor diagnostic row per training run:

```bash
results/launch_logs/ablation/<server2_run>/rdi_nn_match.csv
```

The CSV includes the row label, option switches, effective NN representation,
and `nn_match_percent`.

### Server3: AGDA + Progressive + SL-PPO Advantage

Phase 1 launches PPO architecture/progressive jobs:

```bash
cd /data/Maojie/CaliRoute
bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh ppo
```

Phase 2 launches SL-PPO progressive and advantage-component jobs:

```bash
cd /data/Maojie/CaliRoute
bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh offline
```

PPO jobs:

| Tag | Method | Main knobs |
| --- | --- | --- |
| `s3_agda_kv_only` | PPO | DDE on, QKV delta on KV only |
| `s3_agda_action_key_only` | PPO | DDE on, action key + action bias |
| `s3_agda_both` | PPO | DDE on, KV delta + action key/bias |
| `s3_progressive_rdi_off_agda_on_ppo` | PPO | RDI off, AGDA on |

Offline jobs:

| Tag | Method | Main knobs |
| --- | --- | --- |
| `s3_progressive_rdi_off_agda_off_slppo` | SL-PPO | RDI off, AGDA off |
| `s3_progressive_rdi_on_agda_off_slppo` | SL-PPO | RDI on, AGDA off |
| `s3_progressive_rdi_off_agda_on_slppo` | SL-PPO | RDI off, AGDA on |
| `s3_slppo_group_only` | SL-PPO | group advantage only |
| `s3_slppo_reference_only` | SL-PPO | reference advantage only |

### Server4: n_traj + AGDA Feature Groups

Phase 1 launches PPO feature-group ablations:

```bash
cd /data/Maojie/CaliRoute
bash scripts/ablation/server4_ntraj_feature_groups_3gpu.sh ppo
```

Phase 2 launches SL-PPO `n_traj` ablations:

```bash
cd /data/Maojie/CaliRoute
bash scripts/ablation/server4_ntraj_feature_groups_3gpu.sh offline
```

PPO jobs:

| Tag | Method | Main knobs |
| --- | --- | --- |
| `s4_without_distance_features` | PPO | drop AGDA distance feature group |
| `s4_without_capacity_features` | PPO | drop AGDA capacity feature group |
| `s4_without_battery_features` | PPO | drop AGDA battery feature group |

Offline jobs:

| Tag | Method | Main knobs |
| --- | --- | --- |
| `s4_ntraj100` | SL-PPO | `n_traj=100`, launched first because it is the OOM-risk cell |
| `s4_ntraj5` | SL-PPO | `n_traj=5` |
| `s4_ntraj15` | SL-PPO | `n_traj=15` |

## Nohup Templates

The server scripts detach the whole stage with `nohup` by default. The command
returns after writing the detached launcher pid and log path.

```bash
cd /data/Maojie/CaliRoute
bash scripts/ablation/server2_rdi_4gpu.sh ppo
```

For a two-stage server, launch PPO first and launch offline after the PPO stage
has finished:

```bash
cd /data/Maojie/CaliRoute
bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh ppo
```

```bash
cd /data/Maojie/CaliRoute
bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh offline
```

Each detached stage writes:

```bash
results/launch_logs/ablation/<script_tag>_seed<seed>_<timestamp>/launcher.log
results/launch_logs/ablation/<script_tag>_seed<seed>_<timestamp>/launcher.pid
```

Use `--foreground` for short sanity checks or debugging:

```bash
EPOCHS=1 EVAL_INTERVAL=0 GPU_LIST=0 bash scripts/ablation/server2_rdi_4gpu.sh ppo --foreground
```

## Common Overrides

All server scripts source `common_ablation.sh`, so these overrides work for
every server.

Use a non-default data path:

```bash
DATA_ROOT=/data/Maojie/Routing-D bash scripts/ablation/server2_rdi_4gpu.sh ppo
```

Use specific GPUs:

```bash
GPU_LIST=0,1,2 bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh ppo
```

Use an explicit PPO initial checkpoint:

```bash
INIT_CKPT=/path/to/checkpoint_epoch_0100.pt bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh offline
```

Run fewer epochs for debugging:

```bash
EPOCHS=20 EVAL_INTERVAL=5 bash scripts/ablation/server2_rdi_4gpu.sh ppo
```

Change the seed:

```bash
SEED=1234 bash scripts/ablation/server2_rdi_4gpu.sh ppo
```

Write launch logs to a custom directory:

```bash
LOG_ROOT=/data/Maojie/ablation_logs bash scripts/ablation/server4_ntraj_feature_groups_3gpu.sh offline
```

Use a fixed run directory:

```bash
RUN_DIR=/data/Maojie/ablation_logs/server2_test bash scripts/ablation/server2_rdi_4gpu.sh ppo
```

Tighten or relax the GPU-memory kill threshold:

```bash
MAX_GPU_MEM_MIB=10500 bash scripts/ablation/server4_ntraj_feature_groups_3gpu.sh offline
```

## Default Experiment Geometry

The default EVRPTW Cus100 settings are:

```bash
PROBLEM=evrptw
CUSTOMERS=100
CS=20
SEED=3009
EPOCHS=1500
INIT_EPOCH=100
INIT_EVAL_INTERVAL=20
NUM_ENVS=24
N_TRAJ=50
EVAL_N_TRAJ=50
ROLLOUT_STEPS=160
EVAL_MAX_STEPS=160
PPO_STEP_CHUNK_SIZE=16
NUM_MINIBATCHES=4
EVAL_INTERVAL=20
EVAL_BATCH_SIZE=128
CHECKPOINT_INTERVAL=50
MAX_GPU_MEM_MIB=11000
```

This geometry is intended to fit a 2080Ti. In one-epoch sanity checks, the
largest cell, Server4 `n_traj=100`, exceeded the 11GB budget at
`NUM_ENVS=64` and `NUM_ENVS=32`, but completed at `NUM_ENVS=24` with about
9.1GB peak delta above baseline.

## Initial Checkpoint Logic

The scripts use this priority order for the PPO epoch-100 initial checkpoint:

1. Explicit `INIT_CKPT=/path/to/checkpoint_epoch_0100.pt`.
2. Local CaliRoute checkpoint from the `init` stage:

```bash
results/checkpoints/Cus_100_CS_20/CALIROUTE_EVRPTW_CUS100_CS20_PPO_INIT_SEED3009_E100_N24_R160_2080TI/seed_3009/checkpoint_epoch_0100.pt
```

3. Legacy sibling-repo checkpoint when it exists:

```bash
../EVRPTW-OFFLINE2ONLINE/results/checkpoints/Cus_100_CS_20/O2O_CUS100_PPO_ROUTE_POS_SEED3009_E1500_N128_CHUNK32_EVAL20/seed_3009/checkpoint_epoch_0100.pt
```

If no checkpoint exists and the stage is `init`, `ppo`, or `all`, the script
trains the local initial checkpoint first. Initial PPO training always evaluates
every 20 epochs. If the stage is `offline`, the script waits until the
checkpoint file appears.

## Logs and Monitoring

Launch logs are written under:

```bash
results/launch_logs/ablation/<script_tag>_seed<seed>_<timestamp>/
```

Each job writes:

```bash
<tag>.log
<tag>.gpu_mem.log
<tag>.pid
```

Useful monitoring commands:

```bash
nvidia-smi
tail -f results/launch_logs/ablation/<run_dir>/<tag>.log
tail -f results/launch_logs/ablation/<run_dir>/<tag>.gpu_mem.log
```

The GPU-memory monitor kills a job if memory delta exceeds
`MAX_GPU_MEM_MIB`.

## Values Reused Rather Than Re-run

The scripts only launch the 33 new ablation jobs from the specification. These
table cells should be filled from existing rows:

- AGDA `neither`: reuse Server2 `encoder_bias_only`.
- Progressive `RDI off, AGDA off, PPO`: reuse Server2 `base`.
- Progressive `RDI on, AGDA off, PPO`: reuse Server2 `encoder_bias_only`.
- Progressive `RDI on, AGDA on, PPO`: reuse Server3 `agda_action_key_only`
  or the main-result PPO row.
- Progressive `RDI on, AGDA on, SL-PPO`: reuse the main-result SL-PPO row.
- SL-PPO advantage `both`: reuse the main-result SL-PPO row.
- AGDA feature-group full/no-removal: reuse Server3 `agda_action_key_only`.
- `n_traj` main K value: reuse the main-result SL-PPO row.

## Sanity-Check Templates

Before launching a full server batch, run a short test on one server:

```bash
cd /data/Maojie/CaliRoute
EPOCHS=1 EVAL_INTERVAL=0 GPU_LIST=0 bash scripts/ablation/server2_rdi_4gpu.sh ppo
```

For offline memory risk, test Server4 `n_traj=100` first:

```bash
cd /data/Maojie/CaliRoute
EPOCHS=1 EVAL_INTERVAL=0 GPU_LIST=0 bash scripts/ablation/server4_ntraj_feature_groups_3gpu.sh offline
```

If this exceeds the memory budget, reduce `NUM_ENVS` before launching the full
batch.
