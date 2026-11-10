"""Build negative pseudo-triplets by pairing a positive source with a negative one.

Two mirror modes share one logic: a *positive* file records peptide–X pairs that
bind (label=1), and a *negative* file records peptide–Y pairs that do NOT bind
(label=0). Joining on ``peptide`` yields triplets (peptide, tcr, hla) that are
negative on the full pHLA-TCR complex (PHT=0).

``--mode phla`` (builds ``train_ph_1.csv``)
    Positive source: ``data/data_triple/positive/train_phla_positive.csv``
    (peptide, HLA, label=1 — peptide binds HLA). Negative source: the TCR
    training folds ``data/data_TCR/train_fold_*.csv`` (tcr, peptide, label=0 —
    TCR does not bind the peptide). Labels: PH=1, PT=0, PHT=0.

``--mode ptcr`` (builds ``train_pt_1.csv``)
    Positive source: ``data/data_triple/positive/train_ptcr_positive.csv``
    (peptide, tcr, label=1 — TCR binds the peptide). Negative source: the HLA
    training folds ``data/data_HLA/train_fold_*.csv`` (peptide, HLA, label=0 —
    peptide does not bind HLA). Labels: PH=0, PT=1, PHT=0.

For every negative row whose peptide appears in the positive source, the
peptide is paired with **every** positive partner recorded for it, producing
deduped triplets (peptide, tcr, hla). Output columns:
``peptide,tcr,hla,PH,PT,PHT``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TripletSpec = tuple[Path, str, str, str, int, int, int]


def spec_for(mode: str, split: str) -> TripletSpec:
    """Return (positive_file, pos_partner_col, negative_glob, neg_partner_col, PH, PT, PHT).

    ``mode=phla``  : positive = phla_positive (peptide binds HLA) + TCR negatives
    ``mode=ptcr``  : positive = ptcr_positive (TCR binds peptide) + HLA negatives
    ``split=train``: positive from ``positive/train_*_positive.csv``,
                    negatives from ``data_TCR/data_HLA/train_fold_*.csv``.
    ``split=val``  : same but ``val_*_positive.csv`` and ``val_fold_*.csv``.
    """
    if mode not in ("phla", "ptcr") or split not in ("train", "val"):
        raise ValueError(f"unknown mode={mode!r} or split={split!r}")
    pos_name = f"{split}_{'phla' if mode == 'phla' else 'ptcr'}_positive.csv"
    pos_file = REPO_ROOT / "data/data_triple/positive" / pos_name
    neg_root = "data/data_TCR" if mode == "phla" else "data/data_HLA"
    neg_glob = f"{neg_root}/{split}_fold_*.csv"
    if mode == "phla":
        # positive partner = HLA (binds), negative partner = TCR (does not bind)
        return pos_file, "HLA", neg_glob, "tcr", 1, 0, 0
    # mode == "ptcr": positive partner = TCR (binds), negative partner = HLA (does not bind)
    return pos_file, "tcr", neg_glob, "HLA", 0, 1, 0


def load_positive(path: Path, partner_col: str) -> dict[str, set[str]]:
    """Map peptide -> set of binding partners from the positive file."""
    peptide_to_partners: dict[str, set[str]] = defaultdict(set)
    checked = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"peptide", partner_col, "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            label = str(row["label"]).strip()
            if label != "1":
                raise ValueError(
                    f"{path} contains a non-positive label ({label}); "
                    "expected every positive row to be label=1."
                )
            peptide_to_partners[str(row["peptide"]).strip()].add(
                str(row[partner_col]).strip()
            )
            checked += 1
    print(
        f"[pos ] read {checked} positive rows -> "
        f"{len(peptide_to_partners)} unique peptides"
    )
    return dict(peptide_to_partners)


def collect_negatives(
    fold_paths: list[Path],
    peptide_set: set[str],
    neg_partner_col: str,
) -> list[tuple[str, str]]:
    """Collect (peptide, negative_partner) pairs (label=0, peptide matches)."""
    pairs: set[tuple[str, str]] = set()
    total_neg = 0
    total_rows = 0
    for path in fold_paths:
        fold_neg = 0
        fold_match = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"peptide", neg_partner_col, "label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} must contain columns {sorted(required)}")
            for row in reader:
                total_rows += 1
                if str(row["label"]).strip() != "0":
                    continue
                fold_neg += 1
                peptide = str(row["peptide"]).strip()
                if peptide in peptide_set:
                    fold_match += 1
                    pairs.add((peptide, str(row[neg_partner_col]).strip()))
        total_neg += fold_neg
        print(
            f"[neg ] {path.name}: {fold_neg} label-0 rows, "
            f"{fold_match} matched a positive peptide"
        )
    print(
        f"[neg ] total rows={total_rows}, total label-0={total_neg}, "
        f"deduped matching (peptide,partner) pairs={len(pairs)}"
    )
    return list(pairs)


def write_triplets(
    out_path: Path,
    positive_partners: dict[str, set[str]],
    pairs: list[tuple[str, str]],
    pos_partner_col: str,
    ph: int,
    pt: int,
    pht: int,
) -> int:
    """Expand pairs to (peptide, tcr, hla) triplets and write labeled rows.

    The positive partner is the entity that *binds* the peptide; the negative
    partner is the entity that does not. The output always uses canonical
    columns (peptide, tcr, hla): in ``phla`` mode the positive partner is HLA
    and the negative partner is TCR; in ``ptcr`` mode it is the reverse.
    """
    seen: set[tuple[str, str, str]] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["peptide", "tcr", "hla", "PH", "PT", "PHT"])
        for peptide, neg_partner in pairs:
            for pos_partner in positive_partners[peptide]:
                if pos_partner_col == "HLA":
                    tcr, hla = neg_partner, pos_partner
                else:  # positive partner is TCR
                    tcr, hla = pos_partner, neg_partner
                key = (peptide, tcr, hla)
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow([peptide, tcr, hla, ph, pt, pht])
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["phla", "ptcr"],
        required=True,
        help="phla: positive=phla_positive + TCR-negatives; "
        "ptcr: positive=ptcr_positive + HLA-negatives.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Which split's positive file and folds to draw from.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pos_path, pos_col, neg_glob, neg_col, ph, pt, pht = spec_for(args.mode, args.split)
    fold_paths = sorted(REPO_ROOT.glob(neg_glob))
    if not fold_paths:
        raise FileNotFoundError(f"No negative folds matched: {neg_glob}")
    suffix = "ph_1" if args.mode == "phla" else "pt_1"
    out_path = args.output or REPO_ROOT / "data/data_triple" / f"{args.split}_{suffix}.csv"
    print(f"[init] mode={args.mode} positive={pos_path}")
    print(f"[init] folds={[str(p.relative_to(REPO_ROOT)) for p in fold_paths]}")
    print(f"[init] output={out_path}")

    positive_partners = load_positive(pos_path, pos_col)
    pairs = collect_negatives(fold_paths, set(positive_partners), neg_col)
    n = write_triplets(
        out_path, positive_partners, pairs, pos_col, ph, pt, pht
    )

    uniq_pep = len({p for p, _ in pairs})
    print(f"[done] wrote {n} negative triplets to {out_path}")
    print(f"[done] unique peptides={uniq_pep}")


if __name__ == "__main__":
    main()
