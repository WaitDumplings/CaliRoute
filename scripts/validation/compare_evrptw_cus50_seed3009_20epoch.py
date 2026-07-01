from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NEW_ROOT = ROOT / "results" / "logs" / "Cus_50_CS_10"
OLD_ROOT = Path("/data/Maojie/EVRPTW-OFFLINE2ONLINE/results/logs/Cus_50_CS_10")
OUT_DIR = ROOT / "results" / "validation" / "evrptw_cus50_seed3009_20epoch"
METHODS = ("ppo", "slppo", "dapg", "awbc")


def _read_last_eval(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    epoch20 = [r for r in rows if str(r.get("epoch", "")).strip() == "20"]
    return epoch20[-1] if epoch20 else rows[-1]


def _find_new(method: str) -> Path | None:
    path = NEW_ROOT / f"CALIROUTE_VALIDATE_EVRPTW_CUS50_{method.upper()}_SEED3009_E20" / "seed_3009" / "eval_log.csv"
    return path if path.exists() else None


def _find_old(method: str) -> Path | None:
    if not OLD_ROOT.exists():
        return None
    tokens = ("cus50", method, "3009")
    candidates = [
        path
        for path in OLD_ROOT.glob("*/seed_3009/eval_log.csv")
        if all(token in path.parts[-3].lower() for token in tokens)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _metric(row: dict[str, str] | None, key: str) -> str:
    if row is None:
        return ""
    value = row.get(key, "")
    return "" if value is None else str(value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "comparison.csv"
    fields = [
        "method",
        "new_eval_path",
        "old_eval_path",
        "new_epoch",
        "new_min_obj",
        "new_min_veh",
        "new_med_obj",
        "new_med_veh",
        "new_fr",
        "old_epoch",
        "old_min_obj",
        "old_min_veh",
        "old_med_obj",
        "old_med_veh",
        "old_fr",
        "delta_min_obj_new_minus_old",
    ]
    rows = []
    for method in METHODS:
        new_path = _find_new(method)
        old_path = _find_old(method)
        new_row = _read_last_eval(new_path) if new_path else None
        old_row = _read_last_eval(old_path) if old_path else None
        delta = ""
        try:
            if new_row and old_row:
                delta = f"{float(_metric(new_row, 'min_obj')) - float(_metric(old_row, 'min_obj')):.6f}"
        except (TypeError, ValueError):
            delta = ""
        rows.append(
            {
                "method": method,
                "new_eval_path": str(new_path or ""),
                "old_eval_path": str(old_path or ""),
                "new_epoch": _metric(new_row, "epoch"),
                "new_min_obj": _metric(new_row, "min_obj"),
                "new_min_veh": _metric(new_row, "min_veh"),
                "new_med_obj": _metric(new_row, "med_obj"),
                "new_med_veh": _metric(new_row, "med_veh"),
                "new_fr": _metric(new_row, "fr"),
                "old_epoch": _metric(old_row, "epoch"),
                "old_min_obj": _metric(old_row, "min_obj"),
                "old_min_veh": _metric(old_row, "min_veh"),
                "old_med_obj": _metric(old_row, "med_obj"),
                "old_med_veh": _metric(old_row, "med_veh"),
                "old_fr": _metric(old_row, "fr"),
                "delta_min_obj_new_minus_old": delta,
            }
        )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
