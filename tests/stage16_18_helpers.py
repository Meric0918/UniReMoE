"""Small deterministic encoders plus the real residue-level binding network."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from plm_unified.config import ModelConfig
from plm_unified.stage16_config import load_stage16_config
from plm_unified.stage16_data import PairRows
from plm_unified.stage16_checkpoint import digest
from plm_unified.stage16_evaluation import evaluate_all
from plm_unified.stage16_selection import PT_SPLITS
from plm_unified.stage17_model import Stage17Model
from plm_unified.stage8_model import Stage8BindingModel
from plm_unified.stage8_data import Stage8Record


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.dropout = nn.Dropout(0.2)
        self._active_training = False
        self.hidden_size = 8

    def transformer_blocks(self):
        return self.blocks

    def set_active_training(self, value):
        self._active_training = value
        self.train(value)

    def forward(self, sequences):
        ids = torch.tensor(
            [[ord(c) % 30 for c in s] for s in sequences],
            device=self.embedding.weight.device,
        )
        hidden = self.embedding(ids)
        for block in self.blocks:
            hidden = self.dropout(block(hidden))
        return SimpleNamespace(
            hidden=hidden,
            mask=torch.ones(ids.shape, dtype=torch.bool, device=ids.device),
        )


def pairs(task="ptcr", name="train", n=24):
    dataset = object.__new__(PairRows)
    dataset.task = task
    dataset.path = Path("PT" if task == "ptcr" else "PH") / (name + ".csv")
    dataset.file_sha256 = digest([task, name])
    peptides = ("ACDE", "CDEF", "DEFG", "EFGH")
    receptors = ("IKLM", "KLMN", "LMNP", "MNPQ")
    dataset.rows = [(peptides[i % 4], receptors[i % 4], i % 2) for i in range(n)]
    dataset.row_ids = [f"{task}-{name}-{i}" for i in range(n)]
    dataset.keys = [digest([p, r]) for p, r, _ in dataset.rows]
    dataset.manifest = {
        "samples": n,
        "positive_samples": n // 2,
        "negative_samples": n // 2,
        "row_ids": dataset.row_ids,
        "path": str(dataset.path),
    }
    return dataset


def fixture(stage=16, condition=None, *, dropout=0.1, diagnostics=False):
    torch.manual_seed(123)
    c = load_stage16_config(
        Path(__file__).resolve().parents[1] / f"configs/plm_stage{stage}.yaml"
    )
    if condition:
        c.condition = condition
    c.runtime.device = "cpu"
    c.model = ModelConfig(
        peptide_input_dim=8,
        hla_input_dim=8,
        tcr_input_dim=8,
        adapter_hidden_dim=8,
        model_dim=8,
        num_attention_heads=2,
        feedforward_dim=16,
        adapter_dropout=dropout,
        attention_dropout=dropout,
        classifier_dropout=dropout,
        peptide_min_length=4,
        peptide_max_length=4,
        hla_max_length=4,
        tcr_max_length=4,
    )
    c.training.rounds = 2
    c.training.gradient_accumulation_steps = 2
    c.training.mixed_precision = "none"
    c.training.eval_batch_size = 8
    c.training.gradient_diagnostics = diagnostics
    c.training.checkpoint_every_steps = 0
    c.training.monitor_per_split = 4
    m = Stage17Model(
        Stage8BindingModel(c.model), ToyEncoder(), ToyEncoder(), ToyEncoder()
    )
    train_pairs = {task: pairs(task) for task in ("phla", "ptcr")}
    pt_val = {s: pairs(name=s, n=8) for s in PT_SPLITS}
    records = [
        Stage8Record(
            "ACDE", "IKLM", "PQRS", int(k[0]), int(k[1]), int(k[2]), i, "toy", "val"
        )
        for i, k in enumerate(["000", "010", "100", "111"] * 4)
    ]
    d = {
        "pairs": train_pairs,
        "pt_val": pt_val,
        "ph_val": {},
        "joint_train": records,
        "joint_val": records,
        "manifest": {"fixture": "test-only", "version": 1},
    }
    reference = deepcopy(evaluate_all(m, d, c)[0])
    return c, m, d, reference
