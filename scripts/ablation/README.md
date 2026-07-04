# EVRPTW Cus100 Ablation Launch Scripts

Run these from the CaliRoute repository root on branch `ablation`:

```bash
cd /data/Maojie/CaliRoute
git checkout ablation
git pull
```

The scripts detach under `nohup` by default. Add `--foreground` or `--no-detach`
for a local sanity check.

## Initial PPO Checkpoint

SL-PPO ablations and init-checkpoint PPO ablations require the seed-matched PPO
100-epoch initial weight. They do not silently train it. Create or verify it with:

```bash
bash scripts/ablation/init_ppo100_1gpu.sh 0
```

Default checkpoint path:

```bash
results/checkpoints/Cus_100_CS_20/CALIROUTE_EVRPTW_CUS100_CS20_PPO_INIT_SEED3009_E100_N24_R160_2080TI/seed_3009/checkpoint_epoch_0100.pt
```

If that checkpoint is missing, the new SL-PPO/server7 scripts exit with an error
and tell you to run `init_ppo100_1gpu.sh` first.

On 2080Ti, PPO init is memory-bound. The init shell uses
`PPO_STEP_CHUNK_SIZE=16` by default even when the ablation default is larger.
If it is still too close to the limit, run with `INIT_PPO_STEP_CHUNK_SIZE=8` or
lower `NUM_ENVS` for the init run.

## Server Assignment

| Server | Script | Main input | Jobs to run |
| --- | --- | --- | ---: |
| Init | `init_ppo100_1gpu.sh` | `GPU_ID` | 1 |
| Server2 | `server2_rdi_4gpu.sh` | `GPU_ID RDI_OPTION RDI_EMBEDDING RDI_ENCODER_SINKHORN RDI_ENCODER_BIAS` | 8 RDI rows |
| Server3 | `server3_agda_progressive_slppo_4gpu.sh` | `GPU_ID DYNAMIC_KV DYNAMIC_ACTION_LOGITS` | 3 AGDA rows; neither reused |
| Server4 | `server4_expert_budget_incumbent_4gpu.sh` | `GPU_ID EXPERT_BUDGET_S INCUMBENT` | 10 budget x incumbent rows |
| Server5 | `server5_advantage_terms_slppo.sh` | `GPU_ID GROUP_USE REFERENCE_USE` | 2 advantage rows; both reused |
| Server6 | `server6_ntraj_slppo.sh` | `[GPU_ID] N_TRAJ` | 3 K rows; main K reused |
| Server7 | `server7_agda_feature_groups_ppo.sh` | `GPU_ID DISTANCE_FEATURE CAPACITY_FEATURE BATTERY_FEATURE` | 3 feature rows; all-on reused |

## Server2: RDI

Full RDI table:

```bash
bash scripts/ablation/server2_rdi_4gpu.sh 0,1,2,3 all
```

Single rows:

```bash
bash scripts/ablation/server2_rdi_4gpu.sh 0 base false false false
bash scripts/ablation/server2_rdi_4gpu.sh 1 euclidean false false false
bash scripts/ablation/server2_rdi_4gpu.sh 2 graph true false true
```

`RDI_OPTION=all` expands to base, euclidean, SVD-only, Sinkhorn-only,
encoder-bias-only, and the three combinations. RDI rows are plain PPO from
scratch with AGDA disabled. The script writes NN-match diagnostics to
`rdi_nn_match.csv` in the launch directory.

## Server3: AGDA

Fixed: road graph distance, RDI encoder bias only, DDE/query on. Inputs control
`dynamic_kv` and `dynamic_action_logits`:

```bash
bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh 0 true true    # both
bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh 1 true false   # dynamic KV only
bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh 2 false true   # dynamic action logits only
bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh 3 false false  # reuse Server2 encoder_bias_only
```

`dynamic_kv=true` maps to `--qkv-delta kv`. `dynamic_action_logits=true` maps to
`--action-key on --action-bias on`.

## Server4: Expert Budget x Incumbent

Fixed: full architecture, SL-PPO, EVRPTW Cus100, single seed, RDI encoder bias,
AGDA action-key logits. Variables are expert budget and incumbent:

```bash
bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 0 60 true
bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 1 60 false
bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 0 300 true
bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 1 300 false
bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 0 900 true
bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 1 900 false
bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 0 3600 true
bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 1 3600 false
bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 0 7200 true
bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 1 7200 false
```

`incumbent=true` uses `C_ref = min(C_expert@budget, C_incumbent)` when memory is
available. `incumbent=false` uses only `C_expert@budget` and drops the reference
term for instances without an expert at that budget.

Training logs include both static expert coverage and dynamic SL-PPO reference
coverage:

```text
expert_reference_coverage      # expert trace coverage at the selected budget
sl_reference_coverage          # expert or memory incumbent available in that batch
sl_expert_reference_coverage
sl_memory_reference_coverage
```

Final validation cost is in `eval_log.csv` and the matching final-epoch row of
`train_log.csv`.

## Server5: Advantage Terms

Fixed: full architecture, SL-PPO, expert budget 7200s, incumbent on. Only the two
advantage switches vary:

```bash
bash scripts/ablation/server5_advantage_terms_slppo.sh 0 true false   # group_only
bash scripts/ablation/server5_advantage_terms_slppo.sh 1 false true   # reference_only
bash scripts/ablation/server5_advantage_terms_slppo.sh 2 true true    # reuse Main Results SL-PPO
```

`false false` is outside this ablation and is rejected.

## Server6: Training K / N_TRAJ

Fixed: full architecture, SL-PPO, expert budget 7200s, incumbent on. Eval width is
always forced to `eval_n_traj=50`; only training `n_traj` changes.

Run K=100 first on 2080Ti because it is the OOM-risk cell:

```bash
bash scripts/ablation/server6_ntraj_slppo.sh 0 100
bash scripts/ablation/server6_ntraj_slppo.sh 1 5
bash scripts/ablation/server6_ntraj_slppo.sh 2 15
```

You can also omit GPU and use GPU 0 by default:

```bash
bash scripts/ablation/server6_ntraj_slppo.sh 60
```

## Server7: AGDA Feature Groups

Fixed: full architecture, plain PPO, EVRPTW Cus100, single seed, RDI encoder
bias, AGDA action-key logits. Set exactly one feature group to `false`:

```bash
bash scripts/ablation/server7_agda_feature_groups_ppo.sh 0 false true true  # without_distance
bash scripts/ablation/server7_agda_feature_groups_ppo.sh 1 true false true  # without_capacity
bash scripts/ablation/server7_agda_feature_groups_ppo.sh 2 true true false  # without_battery
bash scripts/ablation/server7_agda_feature_groups_ppo.sh 3 true true true   # reuse Server3 action_key_only
```

Multiple `false` values are outside this ablation and are rejected.

## Common Overrides

Defaults are EVRPTW Cus100, CS20, seed 3009, 1500 epochs, eval every 20 epochs,
`num_envs=24`, `n_traj=50`, `eval_n_traj=50`, rollout steps 160, and mixed
precision.

Useful overrides:

```bash
SEED=1234 bash scripts/ablation/init_ppo100_1gpu.sh 0
DATA_ROOT=/data/Maojie/Routing-D bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 0 7200 true
EPOCHS=1 EVAL_INTERVAL=0 bash scripts/ablation/server7_agda_feature_groups_ppo.sh 0 false true true --foreground
LOG_ROOT=/data/Maojie/ablation_logs bash scripts/ablation/server6_ntraj_slppo.sh 0 100
MAX_GPU_MEM_MIB=10500 bash scripts/ablation/server6_ntraj_slppo.sh 0 100
INIT_CKPT=/path/to/checkpoint_epoch_0100.pt bash scripts/ablation/server5_advantage_terms_slppo.sh 0 true false
```

Each detached launcher writes:

```text
results/launch_logs/ablation/<script_tag>_seed<seed>_<timestamp>/launcher.log
results/launch_logs/ablation/<script_tag>_seed<seed>_<timestamp>/launcher.pid
```

Each job writes `<tag>.log`, `<tag>.gpu_mem.log`, and `<tag>.pid` in the same
launch directory.
