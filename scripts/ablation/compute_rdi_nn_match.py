#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline2online.instance_adapter import iter_adapted_instances


def _coords(instance) -> np.ndarray:
    depot = np.asarray(instance.depot, dtype=np.float64).reshape(1, 2)
    customers = np.asarray(instance.customers, dtype=np.float64)
    stations = np.asarray(instance.charging_stations, dtype=np.float64)
    return np.vstack([depot, customers, stations])


def _euclidean(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _svd_reconstruction(distance: np.ndarray, rank: int) -> np.ndarray:
    safe = np.asarray(distance, dtype=np.float64)
    finite = np.isfinite(safe)
    values = safe[finite]
    if values.size == 0:
        return np.full_like(safe, np.nan)
    normalized = np.where(finite, safe, float(np.mean(values)))
    normalized = (normalized - float(np.mean(values))) / max(float(np.std(values)), 1e-9)
    k = max(1, min(int(rank), normalized.shape[0], normalized.shape[1]))
    u, s, vh = np.linalg.svd(normalized, full_matrices=False)
    u = u[:, :k]
    s = s[:k]
    v = vh[:k, :].T
    q = u * np.sqrt(np.maximum(s, 0.0))[None, :]
    key = v * np.sqrt(np.maximum(s, 0.0))[None, :]
    return q @ key.T


def _nearest(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(mat, np.inf)
    mat[~np.isfinite(mat)] = np.inf
    return np.argmin(mat, axis=1)


def _effective_distance(instance, representation: str, rank: int) -> np.ndarray | None:
    rep = representation.lower().replace("-", "_")
    road = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    if rep in {"none", "base", "off"}:
        return None
    if rep in {"road", "graph", "encoder_bias", "sinkhorn"}:
        return road
    if rep in {"euclidean", "coord", "coordinate"}:
        return _euclidean(_coords(instance))
    if rep in {"svd", "embedding_svd"}:
        return _svd_reconstruction(road, rank)
    raise ValueError(f"unknown representation: {representation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute road-NN match for RDI ablation rows.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--customers", type=int, required=True)
    parser.add_argument("--charging-stations", type=int, default=20)
    parser.add_argument("--problem", default="evrptw")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--distance-option", choices=["base", "euclidean", "graph"], default=None)
    parser.add_argument("--embedding-svd", choices=["true", "false"], default=None)
    parser.add_argument("--encoder-sinkhorn", choices=["true", "false"], default=None)
    parser.add_argument("--encoder-bias", choices=["true", "false"], default=None)
    parser.add_argument("--representation", choices=["none", "base", "euclidean", "svd", "road", "graph"], required=True)
    parser.add_argument("--svd-rank", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    total = 0
    matched = 0
    used_instances = 0
    for idx, instance in enumerate(
        iter_adapted_instances(
            args.dataset,
            num_customers=args.customers,
            num_charging_stations=args.charging_stations,
            problem_type=args.problem,
        )
    ):
        if args.limit is not None and idx >= int(args.limit):
            break
        road = np.asarray(instance.distance_matrix_km, dtype=np.float64)
        effective = _effective_distance(instance, args.representation, args.svd_rank)
        if effective is None:
            continue
        n = min(road.shape[0], effective.shape[0])
        road_nn = _nearest(road[:n, :n])
        eff_nn = _nearest(effective[:n, :n])
        matched += int(np.sum(road_nn == eff_nn))
        total += int(n)
        used_instances += 1

    ratio = float(matched / total) if total else float("nan")
    row = {
        "label": args.label or "",
        "distance_option": args.distance_option or "",
        "embedding_svd": args.embedding_svd or "",
        "encoder_sinkhorn": args.encoder_sinkhorn or "",
        "encoder_bias": args.encoder_bias or "",
        "dataset": str(Path(args.dataset)),
        "problem": args.problem,
        "customers": args.customers,
        "charging_stations": args.charging_stations,
        "representation": args.representation,
        "svd_rank": args.svd_rank,
        "instances_used": used_instances,
        "nodes_compared": total,
        "nn_match_ratio": ratio,
        "nn_match_percent": ratio * 100.0 if math.isfinite(ratio) else float("nan"),
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out.exists() or out.stat().st_size == 0
        with out.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
