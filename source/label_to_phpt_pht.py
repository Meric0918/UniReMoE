"""Replace a triplet file's ``label`` column with ``PH,PT,PHT`` (all = label).

For every row, ``PH = PT = PHT = label`` (1 -> all 1, 0 -> all 0). The file is
rewritten in place with columns ``peptide,tcr,hla,PH,PT,PHT``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def transform(path: Path) -> int:
    records: list[tuple[str, str, str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise ValueError(
                f"{path} must contain a 'label' column; got {reader.fieldnames}"
            )
        for row in reader:
            records.append(
                (
                    str(row["peptide"]).strip(),
                    str(row["tcr"]).strip(),
                    str(row["hla"]).strip(),
                    int(row["label"]),
                )
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["peptide", "tcr", "hla", "PH", "PT", "PHT"])
        for peptide, tcr, hla, label in records:
            writer.writerow([peptide, tcr, hla, label, label, label])
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.files:
        n = transform(path)
        print(f"[done] {path}: rewrote {n} rows -> peptide,tcr,hla,PH,PT,PHT")


if __name__ == "__main__":
    main()
