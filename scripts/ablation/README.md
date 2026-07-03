# EVRPTW Cus100 Ablation Launch Scripts

Each canonical `server*.sh` script is meant to be run once on the
corresponding physical server. The script assigns jobs to local GPUs in waves:
4-GPU scripts use GPU `0,1,2,3`; the 3-GPU script uses GPU `0,1,2`.

| Physical server | Script | GPUs | Runs | Spec group |
| --- | --- | --- | ---: | --- |
| Server1 | `server1_expert_budget_incumbent_4gpu.sh` | `0,1,2,3` | 10 | expert budget x incumbent robustness |
| Server2 | `server2_rdi_4gpu.sh` | `0,1,2,3` | 8 | RDI ablation |
| Server3 | `server3_agda_progressive_slppo_4gpu.sh` | `0,1,2,3` | 9 | AGDA + progressive + SL-PPO advantage |
| Server4 | `server4_ntraj_feature_groups_3gpu.sh` | `0,1,2` | 6 | n_traj + AGDA feature groups |

Example:

```bash
bash scripts/ablation/server2_rdi_4gpu.sh
```

Useful overrides:

```bash
INIT_CKPT=/path/to/checkpoint_epoch_0100.pt bash scripts/ablation/server2_rdi_4gpu.sh
GPU_LIST=0,1,2 bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh
EPOCHS=300 bash scripts/ablation/server1_expert_budget_incumbent_4gpu.sh
```

All scripts source `common_ablation.sh`, which controls shared paths, seed,
2080Ti-safe batch geometry, logging, and GPU-memory monitoring.

The default Cus100 geometry is:

```bash
NUM_ENVS=24
N_TRAJ=50
ROLLOUT_STEPS=160
PPO_STEP_CHUNK_SIZE=16
NUM_MINIBATCHES=4
EVAL_BATCH_SIZE=128
```

This keeps the largest cell, Server4 `n_traj=100`, under the 11GB 2080Ti
budget in a one-epoch GPU sanity check. In that check, `NUM_ENVS=64` and
`NUM_ENVS=32` exceeded the budget for `n_traj=100`, while `NUM_ENVS=24`
peaked at about 9.1GB above baseline.

By default, the scripts reuse the existing EVRPTW Cus100 PPO epoch-100
checkpoint from the sibling `EVRPTW-OFFLINE2ONLINE` repository when it exists.
Set `INIT_CKPT=/path/to/checkpoint_epoch_0100.pt` to override this explicitly.

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
