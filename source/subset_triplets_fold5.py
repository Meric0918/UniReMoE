"""Subset the four pseudo-triplet files to peptides present in a fold's splits.

For each triplet file under ``data/data_triple/`` we keep only rows whose
``peptide`` also appears in the fold-N source data, paired by origin:

* ``*_ph_1.csv`` (PH=1, built from TCR-negatives) -> peptide pool drawn from
  ``data/data_TCR/{train,val}_fold_{N}.csv``.
* ``*_pt_1.csv`` (PT=1, built from HLA-negatives) -> peptide pool drawn from
  ``data/data_HLA/{train,val}_fold_{N}.csv``.

Output: ``data/data_triple/fold{N}/{train_ph_1,train_pt_1,val_ph_1,val_pt_1}.csv``
with the same ``peptide,tcr,hla,PH,PT,PHT`` schema.

Note: with the current data the per-source fold pools fully contain the
triplet peptides, so every row is retained (a no-op snapshot); the membership
filter is still applied so the subset stays correct if inputs change later.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# triplet file -> (pool source dir, label that identifies it)
FILE_SPECS: list[tuple[str, str]] = [
    ("train_ph_1.csv", "data/data_TCR"),  # ph files -> TCR fold-5 pool
    ("val_ph_1.csv", "data/data_TCR"),
    ("train_pt_1.csv", "data/data_HLA"),  # pt files -> HLA fold-5 pool
    ("val_pt_1.csv", "data/data_HLA"),
]


def fold5_peptide_pool(source_dir: Path, fold: int) -> set[str]:
    """Union of peptides in {train,val}_fold_{fold}.csv under source_dir."""
    pool: set[str] = set()
    for split in ("train", "val"):
        path = source_dir / f"{split}_fold_{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        n = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "peptide" not in reader.fieldnames:
                raise ValueError(f"{path} must contain a 'peptide' column")
            for row in reader:
                pool.add(str(row["peptide"]).strip())
                n += 1
        print(f"[pool] {path.relative_to(REPO_ROOT)}: {n} rows")
    return pool


def filter_triplet(src: Path, dst: Path, pool: set[str]) -> tuple[int, int]:
    """Write rows of src whose peptide is in pool to dst. Return (kept, total)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    with src.open("r", encoding="utf-8", newline="") as fin, dst.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"{src} is empty / has no header")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            total += 1
            if str(row["peptide"]).strip() in pool:
                writer.writerow(row)
                kept += 1
    return kept, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        type=int,
        default=5,
        help="Fold number whose {train,val}_fold_{N}.csv files define the pool.",
    )
    parser.add_argument(
        "--triplet-dir",
        type=Path,
        default=REPO_ROOT / "data/data_triple",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to data/data_triple/fold{N}.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or REPO_ROOT / "data/data_triple" / f"fold{args.fold}"

    # Cache pools per source dir.
    pool_cache: dict[Path, set[str]] = {}

    for name, source_rel in FILE_SPECS:
        source_dir = REPO_ROOT / source_rel
        if source_dir not in pool_cache:
            print(f"\n[{name}] building fold-{args.fold} peptide pool from {source_rel}")
            pool_cache[source_dir] = fold5_peptide_pool(source_dir, args.fold)
            print(f"[{name}] pool size: {len(pool_cache[source_dir])} peptides")
        pool = pool_cache[source_dir]
        src = args.triplet_dir / name
        dst = out_dir / name
        kept, total = filter_triplet(src, dst, pool)
        pct = 100.0 * kept / total if total else 0.0
        print(
            f"[{name}] {src.relative_to(REPO_ROOT)} -> "
            f"{dst.relative_to(REPO_ROOT)}: {kept}/{total} kept ({pct:.2f}%)"
        )

    print(f"\n[done] wrote {len(FILE_SPECS)} files to {out_dir}")


if __name__ == "__main__":
    main()
