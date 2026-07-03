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
