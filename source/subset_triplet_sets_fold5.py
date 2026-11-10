"""Subset ``train_set.csv`` / ``val_set.csv`` to peptides present in fold-N data.

``train_set.csv`` is filtered by the union of peptides in
``data/data_HLA/train_fold_{N}.csv`` and ``data/data_TCR/train_fold_{N}.csv``;
``val_set.csv`` by the union of peptides in the two ``val_fold_{N}.csv`` files.
Rows whose ``peptide`` is in the corresponding fold-N pool are written to
``data/data_triple/fold{N}/{train_fold{N},val_fold{N}}.csv`` keeping the original
``peptide,tcr,hla,label`` schema.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (source triplet file, output name template, [fold source dirs], split label)
# output name template uses {N} for the fold number.
SPECS: list[tuple[str, str, list[str], str]] = [
    (
        "train_set.csv",
        "train_fold{N}.csv",
        ["data/data_HLA", "data/data_TCR"],
        "train",
    ),
    (
        "val_set.csv",
        "val_fold{N}.csv",
        ["data/data_HLA", "data/data_TCR"],
        "val",
    ),
]


def fold5_peptide_pool(source_dirs: list[Path], split: str, fold: int) -> set[str]:
    """Union of peptides in {source_dir}/{split}_fold_{fold}.csv across dirs."""
    pool: set[str] = set()
    for source_dir in source_dirs:
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

    for src_name, out_tpl, source_rels, split in SPECS:
        out_name = out_tpl.format(N=args.fold)
        source_dirs = [REPO_ROOT / r for r in source_rels]
        print(f"\n[{out_name}] pool = {split}_fold_{args.fold} across {source_rels}")
        pool = fold5_peptide_pool(source_dirs, split, args.fold)
        print(f"[{out_name}] pool size: {len(pool)} peptides")
        kept, total = filter_triplet(
            args.triplet_dir / src_name, out_dir / out_name, pool
        )
        pct = 100.0 * kept / total if total else 0.0
        print(
            f"[{out_name}] {src_name} -> {out_name}: {kept}/{total} kept ({pct:.2f}%)"
        )

    print(f"\n[done] wrote {len(SPECS)} files to {out_dir}")


if __name__ == "__main__":
    main()
