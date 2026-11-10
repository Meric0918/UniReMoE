"""Deterministic shadow selectors and an explicit PT/PHT feasible region."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

PT_SPLITS = ("val_unseen_pair", "val_unseen_peptide", "val_unseen_tcr")
SCORE_METRICS = ("auroc", "aupr", "accuracy", "mcc", "f1")


def task_score(metrics):
    return sum(float(metrics[k]) for k in SCORE_METRICS) / len(SCORE_METRICS)


def selection_scores(metrics):
    pt, pht = metrics["pt_independent"], metrics["pht_val"]
    if set(pt) != set(PT_SPLITS):
        raise ValueError("Exactly three independent PT validation splits are required")
    legacy = (
        0.4 * task_score(pht["pht"])
        + 0.2 * task_score(pht["phla"])
        + 0.4 * task_score(pht["ptcr"])
    )
    joint = (
        0.6 * sum(task_score(pt[s]) for s in PT_SPLITS) / 3
        + 0.3 * task_score(pht["pht"])
        + 0.1 * task_score(pht["phla"])
    )
    return {"legacy": legacy, "pt_validation": joint, "guarded": joint}


def retention_gate(metrics, reference, policy):
    pt, original = metrics["pt_independent"], reference["pt_independent"]
    if set(pt) != set(PT_SPLITS) or set(original) != set(PT_SPLITS):
        raise ValueError("Missing/extra independent PT split")
    failures, deltas = [], {}
    for split in PT_SPLITS:
        for count in ("samples", "positive_samples", "negative_samples"):
            if pt[split][count] != original[split][count]:
                failures.append(f"{split}:{count}")
    for key in ("auroc", "aupr", "mcc"):
        values = []
        for split in PT_SPLITS:
            current, old = float(pt[split][key]), float(original[split][key])
            if not math.isfinite(current) or not math.isfinite(old):
                failures.append(f"{split}:{key}:nonfinite")
                continue
            delta = current - old
            deltas[f"{split}/{key}"] = delta
            values.append(delta)
            if delta < -policy.split_tolerance:
                failures.append(f"{split}:{key}")
        if len(values) == 3:
            deltas[f"macro/{key}"] = sum(values) / 3
            if sum(values) / 3 < -policy.macro_tolerance:
                failures.append(f"macro:{key}")
    for key in ("samples", "positive_samples", "negative_samples"):
        if metrics["pht_val"]["pht"][key] != reference["pht_val"]["pht"][key]:
            failures.append(f"pht:{key}")
    score, initial = task_score(metrics["pht_val"]["pht"]), task_score(
        reference["pht_val"]["pht"]
    )
    if not math.isfinite(score) or not math.isfinite(initial):
        failures.append("pht:nonfinite")
    else:
        deltas["pht/performance_avg"] = score - initial
        if score < initial - policy.pht_tolerance:
            failures.append("pht:performance_avg")
    for name, value in selection_scores(metrics).items():
        if not math.isfinite(value):
            failures.append(f"score:{name}:nonfinite")
    return {
        "pass": not failures,
        "failures": failures,
        "deltas": deltas,
        "worst_pt_delta": min(
            (v for k, v in deltas.items() if k.startswith("val_")), default=None
        ),
    }


@dataclass
class SelectorState:
    score: float | None = None
    round: int | None = None
    artifact: str | None = None
    sha256: str | None = None
    bad_rounds: int = 0


class ShadowSelectors:
    def __init__(self, policy):
        self.policy = policy
        self.states = {
            k: SelectorState() for k in ("legacy", "pt_validation", "guarded")
        }

    def consider(self, metrics, reference, round_index):
        gate = retention_gate(metrics, reference, self.policy)
        scores = selection_scores(metrics)
        improved = []
        for name, score in scores.items():
            state = self.states[name]
            eligible = math.isfinite(score) and (name != "guarded" or gate["pass"])
            if eligible and (
                state.score is None or score > state.score + self.policy.min_delta
            ):
                state.score, state.round, state.bad_rounds = score, round_index, 0
                improved.append(name)
            elif round_index > 0:
                state.bad_rounds += 1
        return {
            "round": round_index,
            "scores": scores,
            "gate": gate,
            "improved": improved,
        }

    def state_dict(self):
        return {k: asdict(v) for k, v in self.states.items()}

    def load_state_dict(self, states):
        if set(states) != set(self.states):
            raise ValueError("Selector schema mismatch")
        self.states = {k: SelectorState(**v) for k, v in states.items()}
