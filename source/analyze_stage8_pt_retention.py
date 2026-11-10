"""Audit Stage-8 PT supervision and checkpoint drift without model inference.

Only train/validation data are inspected. Test data are deliberately excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from plm_unified.data import OnlinePairDataset
from plm_unified.stage8_config import load_stage8_config
from plm_unified.stage8_data import audit_stage8_class_dirs


def pair_labels(rows):
    result = defaultdict(set)
    for peptide, receptor, label in rows:
        result[peptide, receptor].add(label)
    return result


def describe(rows):
    counts = Counter(p for p, _, _ in rows)
    labels = pair_labels(rows)
    return {
        "rows": len(rows), "positive": sum(y for _, _, y in rows),
        "unique_peptides": len(counts),
        "unique_tcrs": len({t for _, t, _ in rows}),
        "unique_pairs": len(labels),
        "conflicting_pair_keys": sum(len(v) > 1 for v in labels.values()),
        "top_peptides": counts.most_common(10),
    }


def run(config_path, stage1_ptcr):
    config = load_stage8_config(config_path)
    audit = audit_stage8_class_dirs(
        config.data_dir(), class_dirs=config.stage8.class_dirs,
        train_filename=config.stage8.train_filename,
        val_filename=config.stage8.val_filename, model=config.model,
        reject_noncanonical_sequences=config.stage8.reject_noncanonical_sequences,
    )
    audit.require_clean()
    root = config.paths.data_root
    train = OnlinePairDataset(root / "PT/train.csv", "ptcr").rows
    pt_labels = pair_labels(train)
    triples = audit.train_records
    projected = [(r.peptide, r.tcr, r.pt) for r in triples]
    joint_labels = pair_labels(projected)
    train_peptides = {p for p, _, _ in train}
    joint_peptides = {p for p, _, _ in projected}
    joint_tcrs = {t for _, t, _ in projected}
    by_type = {}
    for kind in config.stage8.class_dirs:
        records = [r for r in triples if r.label_type == kind[-3:]]
        rows = [(r.peptide, r.tcr, r.pt) for r in records]
        overlap = [r for r in records if (r.peptide, r.tcr) in pt_labels]
        contradictory = [r for r in overlap if r.pt not in pt_labels[r.peptide, r.tcr]]
        by_type[kind[-3:]] = {
            **describe(rows), "rows_overlapping_pt_train": len(overlap),
            "rows_disagreeing_with_all_pt_train_labels": len(contradictory),
            "peptides_overlapping_pt_train": len({r.peptide for r in records} & train_peptides),
        }
    validation = {}
    for path in sorted((root / "PT/benchmarks").glob("val_*.csv")):
        rows = OnlinePairDataset(path, "ptcr").rows
        overlap = [(p, t, y) for p, t, y in rows if (p, t) in joint_labels]
        validation[path.stem] = {
            **describe(rows), "rows_with_pair_in_pht_train": len(overlap),
            "rows_with_peptide_in_pht_train": sum(p in joint_peptides for p, _, _ in rows),
            "rows_with_tcr_in_pht_train": sum(t in joint_tcrs for _, t, _ in rows),
            "overlap_rows_with_disagreeing_pht_label": sum(y not in joint_labels[p, t] for p, t, y in overlap),
        }
    counts = Counter(r.label_type for r in triples)
    joint_draws = 4 * max(counts.values())
    replay = min(len(train), int(len(triples) * config.stage8.binary_replay_ratio))
    result = {
        "dataset_fingerprint": audit.dataset_fingerprint,
        "pht_audit": audit.report, "pt_train": describe(train),
        "pht_train_projected_pt": describe(projected), "pht_train_by_type": by_type,
        "pt_validation": validation,
        "exposure_per_round": {
            "joint_draws": joint_draws, "pt_replay_draws": replay,
            "pt_train_coverage_fraction": replay / len(train),
            "pt_replay_to_joint_ratio": replay / joint_draws,
        },
    }
    joint = torch.load(config.paths.stage1_checkpoint, weights_only=False, map_location="cpu", mmap=True)
    best_pt = torch.load(stage1_ptcr, weights_only=False, map_location="cpu", mmap=True)
    drift = {}
    for module in ("peptide_adapter", "tcr_adapter", "ptcr_cross_attention", "ptcr_pooling", "ptcr_classifier"):
        pairs = [(v.float(), joint["model"][k].float()) for k, v in best_pt["model"].items() if k.startswith(module + ".")]
        squared_diff = sum(float(((a-b)**2).sum()) for a, b in pairs)
        squared_base = sum(float((a**2).sum()) for a, _ in pairs)
        drift[module] = {
            "equal": all(torch.equal(a, b) for a, b in pairs),
            "relative_l2_to_best_ptcr": (squared_diff / max(squared_base, 1e-30)) ** 0.5,
        }
    result["stage1_sources"] = {
        "joint_path": str(config.paths.stage1_checkpoint), "best_ptcr_path": str(stage1_ptcr),
        "joint_round": joint.get("current_round"), "ptcr_round": best_pt.get("current_round"),
        "joint_current_task_scores": joint.get("current_task_scores"),
        "joint_task_step_limits": joint.get("task_step_limits"),
        "ptcr_best_score": best_pt.get("best_score"), "module_comparison": drift,
        "saved_training_config": best_pt["config"]["training"],
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage1-ptcr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run(args.config, args.stage1_ptcr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
