#!/usr/bin/env python3
"""Live monitor for Stage 1, Stage 3, and Stage 8 training metrics.

The monitor reads the plain-text training logs emitted by the three launchers
and can also read SwanLab's append-only ``backup.swanlab`` and ``run-*.swanlab``
files.  It keeps the
latest row for each stage/phase/task/split and renders separate train and
validation tables.

Examples:

    python source/monitor_training_metrics.py \
        --stage1-log training_logs/stage1.log \
        --stage3-log training_logs/stage3.log \
        --stage8-log training_logs/stage8.log

    python source/monitor_training_metrics.py \
        --input stage1=training_logs/stage1.log \
        --input stage3=training_logs/stage3.log \
        --input stage8=swanlog/PLM_Stage8/run-20260831_222354-example \
        --plot --plot-file outputs/training_metrics.png

    python source/monitor_training_metrics.py \
        --stage8-log swanlog/PLM_Stage8/run-20260901_230137-example/backup.swanlab \
        --once --plot-stage stage8 --plot-metric loss \
        --plot-file figures/stage8/stage8_loss_training_curve.png

Use ``--once`` to print the current snapshot and exit.
Use ``--plot`` for a live window; ``--plot-file`` updates a PNG on every poll.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


STAGE_ORDER = {
    "stage1": 1,
    "stage3": 3,
    "stage8": 8,
    "stage14": 14,
    "stage15": 15,
}
METRIC_NAMES = (
    "loss",
    "clean_loss",
    "adversarial_loss",
    "total_loss",
    "nll",
    "ece",
    "auroc",
    "aupr",
)
NUMBER_PATTERN = (
    r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|"
    r"[-+]?(?:nan|inf))"
)
METRIC_RE = re.compile(
    rf"\b(clean_loss|adversarial_loss|total_loss|loss|nll|ece|auroc|aupr)"
    rf"\s*=\s*({NUMBER_PATTERN})",
    re.IGNORECASE,
)
PROGRESS_VALUE_RE = re.compile(r"=\s*([-+]?\d+(?:\.\d+)?)")
STAGE8_PLAIN_RE = re.compile(
    r"\[METRICS\]\[STAGE8\]\[(?P<phase>[^]]+)]"
    r"\[condition=(?P<condition>[^]]*)]"
    r"\[(?P<progress>[^]]+)]\[step=(?P<step>\d+)]\s+"
    r"(?P<table>TRAIN|VAL)\[(?P<target>[^]]+)]\s+(?P<body>.*)",
    re.IGNORECASE,
)
STAGE8_RUN_RE = re.compile(r"(?P<path>/[^\s]*?/run-[A-Za-z0-9_-]+)")


def canonical_stage(value: str) -> str:
    token = value.strip().lower().replace(" ", "")
    aliases = {
        "1": "stage1", "stage1": "stage1",
        "3": "stage3", "stage3": "stage3",
        "8": "stage8", "stage8": "stage8",
        "14": "stage14", "stage14": "stage14",
        "15": "stage15", "stage15": "stage15",
    }
    try:
        return aliases[token]
    except KeyError as exc:
        raise ValueError(
            f"unknown stage {value!r}; expected stage1, stage3, stage8, stage14, or stage15"
        ) from exc


def normalise_task(value: str) -> str:
    token = value.strip().lower().replace("_", "")
    aliases = {
        "phla": "pHLA",
        "ph": "pHLA",
        "ptcr": "pTCR",
        "pt": "pTCR",
        "pht": "PHT",
        "triplet": "PHT",
    }
    return aliases.get(token, value.strip())


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for match in METRIC_RE.finditer(text):
        try:
            values[match.group(1).lower()] = float(match.group(2))
        except ValueError:
            continue
    return values


def _as_text(value: Any) -> str | None:
    if isinstance(value, (list, tuple)) and value:
        value = value[-1]
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if not isinstance(value, str):
            return None
        try:
            return float(value.strip())
        except ValueError:
            return None
    return float(value)


def _progress(kind: str, value: Any, total: Any = None) -> str:
    number = _as_number(value)
    if number is None:
        return "--"
    rendered = str(int(number)) if number.is_integer() else str(number)
    total_number = _as_number(total)
    if total_number is not None:
        total_rendered = str(int(total_number)) if total_number.is_integer() else str(total_number)
        return f"{kind}={rendered}/{total_rendered}"
    return f"{kind}={rendered}"


def _progress_value(progress: str, step: int | None = None) -> float | None:
    match = PROGRESS_VALUE_RE.search(progress)
    if match:
        return float(match.group(1))
    return float(step) if step is not None else None


@dataclass
class MetricRow:
    source_id: str
    stage: str
    table: str
    run: str
    phase: str
    progress: str
    task: str
    split: str = "-"
    stream: str = ""
    step: int | None = None
    condition: str = ""
    metrics: dict[str, float | None] = field(
        default_factory=lambda: {name: None for name in METRIC_NAMES}
    )


@dataclass
class HistoryPoint:
    progress: str
    x: float | None
    step: int | None
    metrics: dict[str, float | None] = field(
        default_factory=lambda: {name: None for name in METRIC_NAMES}
    )


class MetricStore:
    """Latest metric row per logical stream."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str, str, str, str, str, str], MetricRow] = {}
        self.history: dict[
            tuple[str, str, str, str, str, str, str, str], list[HistoryPoint]
        ] = {}
        self._history_index: dict[
            tuple[str, str, str, str, str, str, str, str],
            dict[tuple[str, int | None], HistoryPoint],
        ] = {}

    def update(
        self,
        *,
        source_id: str,
        stage: str,
        table: str,
        run: str,
        phase: str,
        progress: str,
        task: str,
        split: str = "-",
        stream: str = "",
        step: int | None = None,
        condition: str = "",
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        if table not in {"train", "val"} or not metrics:
            return
        key = (source_id, table, run, phase, condition, task, split, stream)
        row = self.rows.get(key)
        if row is None:
            row = MetricRow(
                source_id=source_id,
                stage=stage,
                table=table,
                run=run,
                phase=phase,
                progress=progress,
                task=task,
                split=split,
                stream=stream,
                step=step,
                condition=condition,
            )
            self.rows[key] = row
        if run:
            row.run = run
        row.progress = progress
        row.step = step
        for name, value in metrics.items():
            if name in row.metrics:
                row.metrics[name] = float(value)

        points = self.history.setdefault(key, [])
        point_index = self._history_index.setdefault(key, {})
        point_key = (progress, None if progress != "--" else step)
        point = point_index.get(point_key)
        if point is None:
            point = HistoryPoint(
                progress=progress,
                x=_progress_value(progress, step),
                step=step,
            )
            point_index[point_key] = point
            points.append(point)
        for name, value in metrics.items():
            if name in point.metrics:
                point.metrics[name] = float(value)

    def table_rows(self, table: str) -> list[MetricRow]:
        return [row for row in self.rows.values() if row.table == table]

    def history_rows(self) -> Iterable[tuple[MetricRow, list[HistoryPoint]]]:
        for key, points in self.history.items():
            row = self.rows.get(key)
            if row is not None:
                yield row, points


@dataclass(frozen=True)
class SourceSpec:
    stage: str
    path: Path


class TextReader:
    """Incrementally parse one plain-text launcher log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending = b""
        self.run = path.stem
        self.stage1_phase = "stage1"
        self.stage1_progress = "--"
        self.stage1_step: int | None = None
        self.stage1_seed = ""
        self.stage3_phase = "stage3"
        self.stage3_progress = "--"
        self.stage3_step: int | None = None
        self.stage3_context: dict[tuple[str, str], tuple[str, int | None]] = {}
        self.pending_swan_path: str | None = None

    def _reset(self) -> None:
        self.offset = 0
        self.pending = b""
        self.stage1_phase = "stage1"
        self.stage1_progress = "--"
        self.stage1_step = None
        self.stage1_seed = ""
        self.stage3_phase = "stage3"
        self.stage3_progress = "--"
        self.stage3_step = None
        self.stage3_context.clear()
        self.pending_swan_path = None

    def poll(self, store: MetricStore, *, stage: str, flush: bool = False) -> list[Path]:
        discovered: list[Path] = []
        if not self.path.is_file():
            return discovered
        try:
            size = self.path.stat().st_size
            if size < self.offset:
                self._reset()
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
            self.offset += len(chunk)
        except OSError:
            return discovered

        data = self.pending + chunk
        lines = data.split(b"\n")
        self.pending = lines.pop()
        for raw_line in lines:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
            discovered.extend(self._parse_line(store, stage, line))
        if flush and self.pending:
            line = self.pending.decode("utf-8", errors="replace").rstrip("\r")
            self.pending = b""
            discovered.extend(self._parse_line(store, stage, line))
        return discovered

    def _update(
        self,
        store: MetricStore,
        *,
        stage: str,
        table: str,
        phase: str,
        progress: str,
        task: str,
        metrics: Mapping[str, float],
        split: str = "-",
        stream: str = "",
        step: int | None = None,
        condition: str = "",
    ) -> None:
        store.update(
            source_id=str(self.path), stage=stage, table=table, run=self.run,
            phase=phase, progress=progress, task=normalise_task(task), split=split,
            stream=stream, step=step, condition=condition, metrics=metrics,
        )

    def _parse_stage8_plain(
        self, store: MetricStore, stage: str, line: str
    ) -> bool:
        match = STAGE8_PLAIN_RE.search(line)
        if not match:
            return False
        table = match.group("table").lower()
        target_parts = [part for part in match.group("target").split("/") if part]
        if table == "train":
            task = normalise_task(target_parts[-1]) if target_parts else "train"
            split = "-"
            stream = target_parts[-2] if len(target_parts) > 1 else ""
        else:
            task = normalise_task(target_parts[-1]) if target_parts else "val"
            split = target_parts[-2] if len(target_parts) > 1 else "val"
            stream = ""
        self._update(
            store, stage=stage, table=table, phase=match.group("phase").lower(),
            progress=match.group("progress"), task=task,
            split=split, stream=stream, step=int(match.group("step")),
            condition=match.group("condition"), metrics=parse_metrics(match.group("body")),
        )
        return True

    def _parse_line(self, store: MetricStore, stage: str, line: str) -> list[Path]:
        discovered: list[Path] = []
        if stage != "stage8":
            if self._parse_stage8_plain(store, stage, line):
                return discovered
            if stage in {"stage1", "stage15"}:
                self._parse_stage1_line(store, line, stage=stage)
            elif stage == "stage3":
                self._parse_stage3_line(store, line)
            return discovered
        continuation = line.strip()
        if self.pending_swan_path and re.fullmatch(r"[A-Za-z0-9_-]+", continuation):
            discovered.append(Path(self.pending_swan_path + continuation) / "backup.swanlab")
            self.pending_swan_path = None
            return discovered
        self.pending_swan_path = None
        for match in STAGE8_RUN_RE.finditer(line):
            run_dir_text = match.group("path").rstrip("/")
            run_dir = Path(run_dir_text)
            backup = run_dir / "backup.swanlab"
            discovered.append(backup)
            if not backup.is_file():
                self.pending_swan_path = run_dir_text

        if self._parse_stage8_plain(store, stage, line):
            return discovered
        if stage == "stage1":
            self._parse_stage1_line(store, line)
        elif stage == "stage3":
            self._parse_stage3_line(store, line)
        return discovered

    def _parse_stage1_line(
        self, store: MetricStore, line: str, *, stage: str = "stage1"
    ) -> None:
        header = re.search(r"\[Stage\s*(?:1|15)([AB])\]", line, re.IGNORECASE)
        if header:
            phase_letter = header.group(1).upper()
            phase_prefix = "stage15" if stage == "stage15" else "stage1"
            self.stage1_phase = f"{phase_prefix}{phase_letter.lower()}"
            progress_match = re.search(
                r"\[(round|epoch)=(\d+)(?:/(\d+))?\]", line, re.IGNORECASE
            )
            if progress_match:
                self.stage1_progress = _progress(
                    progress_match.group(1).lower(),
                    progress_match.group(2),
                    progress_match.group(3),
                )
            step_match = re.search(r"\[step=(\d+)\]", line, re.IGNORECASE)
            if step_match:
                self.stage1_step = int(step_match.group(1))
            seed_match = re.search(r"\[seed=([^]]+)\]", line, re.IGNORECASE)
            if seed_match:
                self.stage1_seed = seed_match.group(1)
                self.run = f"{self.path.stem} seed={self.stage1_seed}"
            if phase_letter == "A":
                task_match = re.search(r"\[task=([^]]+)\]", line, re.IGNORECASE)
                metrics = parse_metrics(line)
                if task_match and metrics:
                    self._update(
                        store, stage=stage, table="train",
                        phase=self.stage1_phase, progress=self.stage1_progress,
                        task=task_match.group(1), step=self.stage1_step, metrics=metrics,
                    )
            return

        match = re.search(r"\b(pHLA|pTCR)\s+([^:]+):\s*(.*)", line, re.IGNORECASE)
        if not match or self.stage1_phase != "stage1b":
            return
        split = match.group(2).strip()
        table = "train" if split.lower() == "train" else "val"
        self._update(
            store, stage=stage, table=table, phase=self.stage1_phase,
            progress=self.stage1_progress, task=match.group(1),
            split="-" if table == "train" else split,
            step=self.stage1_step, metrics=parse_metrics(match.group(3)),
        )

    def _parse_stage3_line(self, store: MetricStore, line: str) -> None:
        marker = re.search(
            r"\[STAGE3([AB])\]\[round=(\d+)\]\[step=(\d+)\]",
            line, re.IGNORECASE,
        )
        if marker:
            self.stage3_phase = f"stage3{marker.group(1).lower()}"
            self.stage3_progress = _progress("round", marker.group(2))
            self.stage3_step = int(marker.group(3))

        start = re.search(
            r"\[TRAIN START\]\[([^]]+)\]\[([^]]+)\]\s+id=(\d+)",
            line, re.IGNORECASE,
        )
        if start:
            phase = start.group(1).lower()
            task = normalise_task(start.group(2))
            kind = "epoch" if phase == "stage3a" else "round"
            progress = _progress(kind, start.group(3))
            self.stage3_phase = phase
            self.stage3_progress = progress
            self.stage3_step = None
            self.stage3_context[(phase, task)] = (progress, None)
            return

        end = re.search(
            r"\[TRAIN END\]\[([^]]+)\]\[([^]]+)\](?P<body>.*)",
            line, re.IGNORECASE,
        )
        if end:
            phase = end.group(1).lower()
            task = normalise_task(end.group(2))
            progress, step = self.stage3_context.get(
                (phase, task), (self.stage3_progress, self.stage3_step)
            )
            self._update(
                store, stage="stage3", table="train", phase=phase,
                progress=progress, task=task, step=step,
                metrics=parse_metrics(end.group("body")),
            )
            return

        multi_val = re.search(
            r"\[VAL END\]\[PHT\]\[([^]]+)\](?P<body>.*)", line, re.IGNORECASE
        )
        if multi_val:
            split = multi_val.group(1).strip()
            body = multi_val.group("body")
            for task_match in re.finditer(
                r"\b(phla|ptcr|pht|triplet)\s*\(([^)]*)\)", body, re.IGNORECASE
            ):
                task = normalise_task(task_match.group(1))
                progress, step = self.stage3_context.get(
                    (self.stage3_phase, task),
                    (self.stage3_progress, self.stage3_step),
                )
                self._update(
                    store, stage="stage3", table="val", phase=self.stage3_phase,
                    progress=progress, task=task, split=split, step=step,
                    metrics=parse_metrics(task_match.group(2)),
                )
            return

        single_val = re.search(
            r"\[VAL END\]\[([^]]+)\](?P<body>.*)", line, re.IGNORECASE
        )
        if single_val:
            task = normalise_task(single_val.group(1))
            progress, step = self.stage3_context.get(
                (self.stage3_phase, task), (self.stage3_progress, self.stage3_step)
            )
            self._update(
                store, stage="stage3", table="val", phase=self.stage3_phase,
                progress=progress, task=task, split="val", step=step,
                metrics=parse_metrics(single_val.group("body")),
            )


class SwanReader:
    """Decode JSON backup records or SwanLab's binary protobuf run records."""

    RECORD_MARKER = b'{"model_type":'

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending = b""
        self.binary_size: int | None = None
        self.run = path.parent.name
        self.stage8_phase = "stage8"
        self.stage8_condition = ""
        self.stage8_progress = "--"
        self.stage8_step: int | None = None
        self.stage14_phase = "stage14"
        self.stage14_condition = ""
        self.stage14_progress = "--"
        self.stage14_step: int | None = None
        self.stage3_progress: dict[str, str] = {}
        self.stage3_step: dict[str, int | None] = {}

    def _reset(self) -> None:
        self.offset = 0
        self.pending = b""
        self.binary_size = None
        self.stage8_phase = "stage8"
        self.stage8_condition = ""
        self.stage8_progress = "--"
        self.stage8_step = None
        self.stage14_phase = "stage14"
        self.stage14_condition = ""
        self.stage14_progress = "--"
        self.stage14_step = None
        self.stage3_progress.clear()
        self.stage3_step.clear()

    @staticmethod
    def _record_value(record: Mapping[str, Any]) -> Any:
        data = record.get("data")
        if isinstance(data, Mapping):
            metric = data.get("metric")
            if isinstance(metric, Mapping) and "data" in metric:
                return metric["data"]
            if "value" in data:
                return data["value"]
        return record.get("value")

    @staticmethod
    def _record_key(record: Mapping[str, Any]) -> str:
        data = record.get("data")
        if isinstance(data, Mapping) and isinstance(data.get("key"), str):
            return data["key"]
        return str(record.get("key", ""))

    @staticmethod
    def _record_step(record: Mapping[str, Any]) -> int | None:
        value = record.get("step")
        if value is None and isinstance(record.get("data"), Mapping):
            value = record["data"].get("step")
        number = _as_number(value)
        return int(number) if number is not None else None

    @staticmethod
    def _record_epoch(record: Mapping[str, Any]) -> float | None:
        value = record.get("epoch")
        if value is None and isinstance(record.get("data"), Mapping):
            value = record["data"].get("epoch")
        return _as_number(value)

    def poll(
        self, store: MetricStore, *, stage: str, flush: bool = False
    ) -> list[Path]:
        if self._is_binary_store():
            return self._poll_binary(store, stage)
        del flush  # SwanLab records are complete JSON lines once newline-terminated.
        if not self.path.is_file():
            return []
        try:
            size = self.path.stat().st_size
            if size < self.offset:
                self._reset()
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
            self.offset += len(chunk)
        except OSError:
            return []

        data = self.pending + chunk
        self.pending = b""
        cursor = 0
        while cursor < len(data):
            start = data.find(self.RECORD_MARKER, cursor)
            if start < 0:
                self.pending = data[max(0, len(data) - len(self.RECORD_MARKER) + 1):]
                break
            end = data.find(b"\n", start)
            if end < 0:
                self.pending = data[start:]
                break
            try:
                record = json.loads(data[start:end].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                cursor = end + 1
                continue
            if isinstance(record, Mapping):
                self._parse_record(store, stage, record)
            cursor = end + 1
        return []

    def _is_binary_store(self) -> bool:
        if not self.path.is_file():
            return False
        try:
            with self.path.open("rb") as handle:
                return handle.read(4) == b":SWL"
        except OSError:
            return False

    def _poll_binary(self, store: MetricStore, stage: str) -> list[Path]:
        """Read the current snapshot of a SwanLab v0.10 protobuf run file.

        The protobuf store is block-framed and may be appended while training is
        running.  Replaying complete records from the beginning keeps context
        keys such as ``stage8/round`` aligned with their metric records and is
        cheap compared with the model training workload.  MetricStore de-duplicates
        history points, so repeated polls do not add duplicate curves.
        """
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if self.binary_size == size:
            return []
        if self.binary_size is not None and size < self.binary_size:
            self._reset()
        self.binary_size = size

        try:
            from swanlab.proto.swanlab.record.v1.record_pb2 import Record
            from swanlab.sdk.internal.core_python.store import DataStoreReader

            reader = DataStoreReader()
            reader.open(self.path)
            try:
                for payload in reader:
                    record = Record()
                    record.ParseFromString(payload)
                    if not record.HasField("scalar"):
                        continue
                    scalar = record.scalar
                    key = scalar.key
                    value = float(scalar.value.number)
                    step = int(scalar.step)
                    if self._parse_context(key, value, step):
                        continue
                    self._parse_scalar_metric(
                        store, stage, key, value, step, epoch=None
                    )
            finally:
                reader.close()
        except Exception:
            # A live run can end with an incomplete final block.  Keep the
            # records already collected and retry the file on the next poll.
            return []
        return []

    def _parse_record(
        self, store: MetricStore, stage: str, record: Mapping[str, Any]
    ) -> None:
        model_type = str(record.get("model_type", ""))
        if model_type == "Experiment":
            data = record.get("data")
            if isinstance(data, Mapping):
                name = data.get("name") or data.get("experiment_name")
            else:
                name = None
            name = name or record.get("name")
            if name:
                self.run = str(name)
            return
        if model_type not in {"Scalar", "Media"}:
            return

        key = self._record_key(record)
        value = self._record_value(record)
        step = self._record_step(record)
        epoch = self._record_epoch(record)
        if self._parse_context(key, value, step):
            return
        if model_type != "Scalar":
            return
        number = _as_number(value)
        if number is None:
            return
        self._parse_scalar_metric(store, stage, key, number, step, epoch)

    def _parse_context(self, key: str, value: Any, step: int | None) -> bool:
        text = _as_text(value)
        if key == "stage8/phase" and text:
            self.stage8_phase = text.lower()
            return True
        if key == "stage8/condition" and text:
            self.stage8_condition = text
            return True
        if key == "stage8/round" and value is not None:
            self.stage8_progress = _progress("round", value)
            self.stage8_step = step
            return True
        if key == "stage8/epoch" and value is not None:
            self.stage8_progress = _progress("epoch", value)
            self.stage8_step = step
            return True
        if key == "stage14/phase" and text:
            self.stage14_phase = text.lower()
            return True
        if key == "stage14/condition" and text:
            self.stage14_condition = text
            return True
        if key == "stage14/round" and value is not None:
            self.stage14_progress = _progress("round", value)
            self.stage14_step = step
            return True
        if key == "stage14/epoch" and value is not None:
            self.stage14_progress = _progress("epoch", value)
            self.stage14_step = step
            return True
        if key.startswith("stage3") and key.endswith("/round"):
            phase = key.split("/", 1)[0].lower()
            self.stage3_progress[phase] = _progress("round", value)
            self.stage3_step[phase] = step
            return True
        if key.startswith("stage3") and key.endswith("/epoch"):
            phase = key.split("/", 1)[0].lower()
            self.stage3_progress[phase] = _progress("epoch", value)
            self.stage3_step[phase] = step
            return True
        return False

    def _update(
        self,
        store: MetricStore,
        *,
        stage: str,
        table: str,
        phase: str,
        progress: str,
        task: str,
        split: str,
        step: int | None,
        condition: str,
        metric: str,
        value: float,
        stream: str = "",
    ) -> None:
        store.update(
            source_id=str(self.path), stage=stage, table=table, run=self.run,
            phase=phase, progress=progress, task=normalise_task(task), split=split,
            stream=stream, step=step, condition=condition, metrics={metric: value},
        )

    def _parse_scalar_metric(
        self,
        store: MetricStore,
        stage: str,
        key: str,
        value: float,
        step: int | None,
        epoch: float | None,
    ) -> None:
        parts = key.lower().split("/")
        raw_metric = parts[-1] if parts else ""
        if stage == "stage14":
            if (
                raw_metric == "total"
                and len(parts) >= 4
                and parts[1:3] == ["train", "joint"]
            ):
                self._parse_stage14_metric(
                    store, parts, "loss", value, step, epoch
                )
            elif raw_metric in METRIC_NAMES:
                self._parse_stage14_metric(
                    store, parts, raw_metric, value, step, epoch
                )
            return
        if not parts or raw_metric not in METRIC_NAMES:
            return
        metric = raw_metric
        if stage in {"stage1", "stage15"}:
            self._parse_stage1_metric(
                store, stage, parts, metric, value, step, epoch
            )
        elif stage == "stage3":
            self._parse_stage3_metric(store, parts, metric, value, step, epoch)
        elif stage == "stage8":
            self._parse_stage8_metric(store, parts, metric, value, step, epoch)
        elif stage == "stage14":
            self._parse_stage14_metric(store, parts, metric, value, step, epoch)

    def _parse_stage1_metric(
        self, store: MetricStore, stage: str, parts: list[str], metric: str,
        value: float, step: int | None, epoch: float | None,
    ) -> None:
        if len(parts) < 4 or parts[0] not in {"warmup", "round"}:
            return
        task = normalise_task(parts[1])
        table = parts[2]
        if table == "train":
            split = "-"
        elif table == "val":
            split = parts[3] if len(parts) >= 5 else "val"
        else:
            return
        phase_prefix = "stage15" if stage == "stage15" else "stage1"
        phase = f"{phase_prefix}a" if parts[0] == "warmup" else f"{phase_prefix}b"
        kind = "epoch" if parts[0] == "warmup" else "round"
        self._update(
            store, stage=stage, table="train" if table == "train" else "val",
            phase=phase, progress=_progress(kind, epoch), task=task, split=split,
            step=step, condition="", metric=metric, value=value,
        )

    def _parse_stage3_metric(
        self, store: MetricStore, parts: list[str], metric: str,
        value: float, step: int | None, epoch: float | None,
    ) -> None:
        if not parts or parts[0] not in {"stage3a", "stage3b"}:
            return
        phase = parts[0]
        progress = self.stage3_progress.get(
            phase, _progress("epoch" if phase == "stage3a" else "round", epoch)
        )
        context_step = self.stage3_step.get(phase, step)
        if "train" in parts:
            index = parts.index("train")
            if index < 2:
                return
            task = parts[index - 1]
            self._update(
                store, stage="stage3", table="train", phase=phase,
                progress=progress, task=task, split="-", step=context_step,
                condition="", metric=metric, value=value,
            )
            return

        # Current Stage-3 validation keys are either
        # stage3x/{split}/{task}/val/{metric} or stage3x/{task}/val[_selected]/{metric}.
        if len(parts) >= 5 and parts[-2] == "val":
            split = parts[1]
            task = parts[2]
        elif len(parts) == 4 and parts[-2] in {"val", "val_selected"}:
            task = parts[1]
            split = parts[-2]
        else:
            return
        self._update(
            store, stage="stage3", table="val", phase=phase, progress=progress,
            task=task, split=split, step=context_step, condition="",
            metric=metric, value=value,
        )

    def _parse_stage8_metric(
        self, store: MetricStore, parts: list[str], metric: str,
        value: float, step: int | None, epoch: float | None,
    ) -> None:
        if len(parts) < 4 or parts[:2] not in (
            ["stage8", "train"], ["stage8", "validation"]
        ):
            return
        progress = self.stage8_progress
        if progress == "--":
            progress = _progress("epoch", epoch)
        current_step = self.stage8_step if self.stage8_step is not None else step
        if parts[1] == "train":
            rest = parts[2:-1]
            if not rest:
                return
            task = rest[-1]
            stream = rest[0] if len(rest) > 1 else "replay"
            self._update(
                store, stage="stage8", table="train", phase=self.stage8_phase,
                progress=progress, task=task, split="-", step=current_step,
                stream=stream, condition=self.stage8_condition,
                metric=metric, value=value,
            )
            return
        task = parts[2]
        self._update(
            store, stage="stage8", table="val", phase=self.stage8_phase,
            progress=progress, task=task, split="val", step=current_step,
            condition=self.stage8_condition, metric=metric, value=value,
        )

    def _parse_stage14_metric(
        self, store: MetricStore, parts: list[str], metric: str,
        value: float, step: int | None, epoch: float | None,
    ) -> None:
        if len(parts) < 4 or parts[0] != "stage14":
            return
        progress = self.stage14_progress
        if progress == "--":
            kind = "epoch" if self.stage14_phase.endswith("a") else "round"
            progress = _progress(kind, epoch)
        current_step = self.stage14_step if self.stage14_step is not None else step
        if parts[1] == "train":
            target = parts[2]
            if target == "joint":
                task = "joint"
                stream = "joint"
            else:
                task = target
                stream = "replay"
            self._update(
                store, stage="stage14", table="train", phase=self.stage14_phase,
                progress=progress, task=task, split="-", step=current_step,
                stream=stream, condition=self.stage14_condition,
                metric=metric, value=value,
            )
            return
        if parts[1] != "validation":
            return
        task = parts[2]
        self._update(
            store, stage="stage14", table="val", phase=self.stage14_phase,
            progress=progress, task=task, split="val", step=current_step,
            condition=self.stage14_condition, metric=metric, value=value,
        )


class Monitor:
    def __init__(self, specs: Iterable[SourceSpec]) -> None:
        self.specs: list[SourceSpec] = []
        self.spec_keys: set[tuple[str, str]] = set()
        self.readers: dict[str, TextReader | SwanReader] = {}
        self.store = MetricStore()
        for spec in specs:
            self.add_spec(spec.stage, spec.path)

    def add_spec(self, stage: str, path: Path) -> None:
        stage = canonical_stage(stage)
        expanded = path.expanduser()
        key = (stage, str(expanded))
        if key in self.spec_keys:
            return
        self.spec_keys.add(key)
        self.specs.append(SourceSpec(stage, expanded))

    @staticmethod
    def _candidate_paths(path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if path.name == "backup.swanlab" or path.suffix == ".swanlab":
            return [path]
        if path.is_dir():
            direct = path / "backup.swanlab"
            if direct.is_file():
                return [direct]
            nested = sorted(path.glob("*/backup.swanlab"))
            if nested:
                return nested
            binary = sorted(path.glob("run-*.swanlab"))
            if binary:
                return binary
            return sorted(path.glob("*.swanlab"))
        return [path]

    @staticmethod
    def _reader_for(path: Path) -> TextReader | SwanReader:
        if path.name == "backup.swanlab" or path.suffix == ".swanlab":
            return SwanReader(path)
        return TextReader(path)

    def poll(self, *, flush: bool = False) -> None:
        index = 0
        while index < len(self.specs):
            spec = self.specs[index]
            index += 1
            for path in self._candidate_paths(spec.path):
                reader_key = str(path.expanduser())
                reader = self.readers.get(reader_key)
                if reader is None:
                    reader = self._reader_for(path)
                    self.readers[reader_key] = reader
                discovered = reader.poll(self.store, stage=spec.stage, flush=flush)
                for backup in discovered:
                    self.add_spec("stage8", backup)


def _shorten(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.4f}"


class LivePlotter:
    """Matplotlib renderer for the history collected by :class:`MetricStore`."""

    PLOT_METRIC_TITLES = {
        "loss": "Loss",
        "clean_loss": "Clean loss",
        "adversarial_loss": "Adversarial loss",
        "total_loss": "Total loss",
        "nll": "NLL",
        "ece": "ECE",
        "auroc": "AUROC",
        "aupr": "AUPR",
    }
    PLOT_METRICS = (
        ("loss", "Loss"),
        ("auroc", "AUROC"),
        ("aupr", "AUPR"),
    )
    PLOT_STAGES = ("stage1", "stage3", "stage8", "stage14", "stage15")
    # Use a separate, high-contrast color for each logical stream.  The old
    # task-only palette made all pHLA validation splits (and all pTCR splits)
    # look nearly identical in a crowded legend.
    SERIES_COLORS = {
        # Stage 1
        ("stage1", "stage1a", "pHLA", "-", "train"): "#264653",
        ("stage1", "stage1a", "pTCR", "-", "train"): "#e76f51",
        ("stage1", "stage1b", "pHLA", "-", "train"): "#2a9d8f",
        ("stage1", "stage1b", "pTCR", "-", "train"): "#f4a261",
        ("stage1", "stage1b", "pHLA", "val_unseen_hla", "val"): "#457b9d",
        ("stage1", "stage1b", "pHLA", "val_unseen_pair", "val"): "#6a4c93",
        ("stage1", "stage1b", "pHLA", "val_unseen_peptide", "val"): "#8338ec",
        ("stage1", "stage1b", "pTCR", "val_unseen_pair", "val"): "#d62828",
        ("stage1", "stage1b", "pTCR", "val_unseen_peptide", "val"): "#f77f00",
        ("stage1", "stage1b", "pTCR", "val_unseen_tcr", "val"): "#ff006e",
        # Stage 3
        ("stage3", "stage3a", "PHT", "-", "train"): "#008000",
        ("stage3", "stage3a", "pHLA", "val", "val"): "#1d4ed8",
        ("stage3", "stage3a", "pTCR", "val", "val"): "#c2410c",
        ("stage3", "stage3a", "PHT", "val", "val"): "#15803d",
        ("stage3", "stage3b", "pHLA", "-", "train"): "#00a6c7",
        ("stage3", "stage3b", "pTCR", "-", "train"): "#fb8500",
        ("stage3", "stage3b", "PHT", "-", "train"): "#06a77d",
        ("stage3", "stage3b", "pHLA", "val", "val"): "#7209b7",
        ("stage3", "stage3b", "pTCR", "val", "val"): "#e63946",
        ("stage3", "stage3b", "PHT", "val", "val"): "#2a9d8f",
    }
    TASK_COLORS = {
        "pHLA": "#0072b2",
        "pTCR": "#d55e00",
        "PHT": "#009e73",
    }
    STAGE8_STREAM_COLORS = {
        ("pHLA", "replay"): "#4c9bd6",
        ("pHLA", "joint"): "#005a9c",
        ("pTCR", "replay"): "#f28e2b",
        ("pTCR", "joint"): "#a83800",
        ("PHT", "joint"): "#009e73",
    }
    STAGE14_STREAM_COLORS = {
        ("pHLA", "replay"): "#4c9bd6",
        ("pTCR", "replay"): "#f28e2b",
        ("joint", "joint"): "#009e73",
    }

    def __init__(
        self,
        *,
        output: Path | None = None,
        show_window: bool = False,
        stages: Iterable[str] | None = None,
        metrics: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Matplotlib is required for plotting; install it with "
                "`pip install matplotlib` or omit --plot"
            ) from exc
        self.plt = plt
        self.output = output.expanduser() if output is not None else None
        self.show_window = show_window
        self.plot_stages = tuple(stages) if stages is not None else self.PLOT_STAGES
        self.plot_metrics = tuple(metrics) if metrics is not None else self.PLOT_METRICS
        if not self.plot_stages or not self.plot_metrics:
            raise ValueError("at least one plot stage and metric are required")
        if self.output is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.show_window:
            self.plt.ion()
        self.figure, self.axes = self.plt.subplots(
            nrows=len(self.plot_metrics),
            ncols=len(self.plot_stages),
            figsize=(18, 11),
            squeeze=False,
        )
        manager = getattr(self.figure.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("PLM Training Metrics")

    @staticmethod
    def _label(row: MetricRow, table: str) -> str:
        parts = [row.task]
        if row.stream:
            parts.append(row.stream)
        if row.split != "-":
            parts.append(row.split)
        parts.append(row.phase)
        if row.condition:
            parts.append(row.condition)
        if row.run:
            parts.append(_shorten(row.run, 24))
        parts.append(table)
        return " / ".join(parts)

    @classmethod
    def _color(cls, row: MetricRow, table: str) -> str:
        if row.stage == "stage8" and row.stream:
            stream_color = cls.STAGE8_STREAM_COLORS.get((row.task, row.stream.lower()))
            if stream_color is not None:
                return stream_color
        if row.stage == "stage14" and row.stream:
            stream_color = cls.STAGE14_STREAM_COLORS.get((row.task, row.stream.lower()))
            if stream_color is not None:
                return stream_color
        color_stage = "stage1" if row.stage == "stage15" else row.stage
        color_phase = row.phase.replace("stage15", "stage1", 1)
        key = (color_stage, color_phase, row.task, row.split, table)
        return cls.SERIES_COLORS.get(key, cls.TASK_COLORS.get(row.task, "#4d4d4d"))

    @staticmethod
    def _direct_label(row: MetricRow, table: str) -> str:
        split_labels = {
            "val_unseen_hla": "val-HLA",
            "val_unseen_pair": "val-pair",
            "val_unseen_peptide": "val-peptide",
            "val_unseen_tcr": "val-TCR",
            "val_selected": "val-selected",
            "val": "val",
        }
        scope = "train" if table == "train" else split_labels.get(row.split, row.split)
        if row.stream:
            scope = f"{scope}/{row.stream}"
        phase = row.phase[-1].upper() if row.phase.startswith("stage") else row.phase
        return f"{row.task} {scope} {phase}"

    @staticmethod
    def _place_direct_labels(
        axis: Any,
        endpoints: list[tuple[str, float, float, str]],
    ) -> None:
        if not endpoints:
            return

        x_values = [item[1] for item in endpoints]
        x_min, x_max = min(x_values), max(x_values)
        x_span = max(x_max - x_min, 1.0)
        x_pad = max(1.0, x_span * 0.10)
        current_left, _ = axis.get_xlim()
        axis.set_xlim(min(current_left, x_min - x_span * 0.03), x_max + x_pad)

        y_low, y_high = axis.get_ylim()
        y_span = max(y_high - y_low, 1e-6)
        lower = y_low + y_span * 0.03
        upper = y_high - y_span * 0.03
        gap = min(y_span * 0.045, (upper - lower) / max(len(endpoints) - 1, 1))

        ordered = sorted(endpoints, key=lambda item: item[2])
        label_y: list[float] = []
        for _, _, y_value, _ in ordered:
            target = min(max(y_value, lower), upper)
            if label_y:
                target = max(target, label_y[-1] + gap)
            label_y.append(target)

        if label_y and label_y[-1] > upper:
            shift = label_y[-1] - upper
            label_y = [value - shift for value in label_y]
        if label_y and label_y[0] < lower:
            if len(label_y) == 1:
                label_y = [(lower + upper) / 2.0]
            else:
                label_y = [
                    lower + index * (upper - lower) / (len(label_y) - 1)
                    for index in range(len(label_y))
                ]

        for (label, x_value, y_value, color), target_y in zip(ordered, label_y):
            text_x = x_value + x_pad * 0.25
            arrowprops = None
            if abs(target_y - y_value) > y_span * 0.01:
                arrowprops = {
                    "arrowstyle": "-",
                    "color": color,
                    "linewidth": 0.8,
                    "alpha": 0.75,
                }
            axis.annotate(
                label,
                xy=(x_value, y_value),
                xytext=(text_x, target_y),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=8,
                color=color,
                annotation_clip=False,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.2,
                },
                arrowprops=arrowprops,
            )

    def update(self, store: MetricStore) -> None:
        for metric_index, (metric, metric_title) in enumerate(self.plot_metrics):
            for stage_index, stage in enumerate(self.plot_stages):
                axis = self.axes[metric_index][stage_index]
                axis.clear()
                axis.set_title(f"{stage.upper()} - {metric_title}")
                axis.grid(True, alpha=0.25)
                if metric != "loss":
                    axis.set_ylim(0.0, 1.05)
                if metric_index == len(self.plot_metrics) - 1:
                    axis.set_xlabel("Epoch / round")
                if stage_index == 0:
                    axis.set_ylabel(metric_title)

                plotted = 0
                endpoints: list[tuple[str, float, float, str]] = []
                for row, points in store.history_rows():
                    if row.stage != stage:
                        continue
                    values = [
                        (point.x, point.metrics.get(metric))
                        for point in points
                        if point.x is not None
                        and point.metrics.get(metric) is not None
                        and math.isfinite(float(point.metrics[metric]))
                    ]
                    if not values:
                        continue
                    values.sort(key=lambda item: item[0])
                    xs = [item[0] for item in values]
                    ys = [float(item[1]) for item in values]
                    color = self._color(row, row.table)
                    table = row.table
                    axis.plot(
                        xs, ys,
                        color=color,
                        linestyle="-" if table == "train" else "--",
                        marker="o" if table == "train" else "s",
                        markersize=3,
                        linewidth=1.4,
                        alpha=0.9,
                        label=self._label(row, table),
                    )
                    endpoints.append(
                        (self._direct_label(row, table), xs[-1], ys[-1], color)
                    )
                    plotted += 1
                if not plotted:
                    axis.text(
                        0.5, 0.5, "waiting for metrics",
                        transform=axis.transAxes, ha="center", va="center",
                        color="#777777",
                    )
                else:
                    self._place_direct_labels(axis, endpoints)

        self.figure.suptitle("PLM Training Metrics: train (solid) vs val (dashed)")
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        self.figure.canvas.draw_idle()
        try:
            self.figure.canvas.flush_events()
        except AttributeError:
            pass
        if self.output is not None:
            self.figure.savefig(self.output, dpi=150)

    def wait(self, seconds: float) -> None:
        if self.show_window:
            self.plt.pause(seconds)
        else:
            time.sleep(seconds)

    def show_once(self) -> None:
        if self.show_window:
            self.plt.ioff()
            self.plt.show(block=True)


def _render_table(title: str, rows: list[MetricRow], max_rows: int) -> None:
    headers = [
        "Stage", "Run/condition", "Phase", "Round/Epoch", "Task/pass", "Split",
        "Loss", "AUROC", "AUPR",
    ]
    caps = [7, 30, 12, 13, 16, 18, 10, 10, 10]
    rows.sort(
        key=lambda row: (
            STAGE_ORDER.get(row.stage, 99), row.run, row.condition,
            row.phase, row.task, row.stream, row.split,
        )
    )
    shown = rows[:max_rows]
    values: list[list[str]] = []
    for row in shown:
        run = row.run or Path(row.source_id).stem
        if row.condition and row.condition not in run:
            run = f"{run}/{row.condition}"
        task = row.task if not row.stream else f"{row.task}/{row.stream}"
        values.append([
            row.stage.replace("stage", "S"), _shorten(run, caps[1]),
            _shorten(row.phase, caps[2]), _shorten(row.progress, caps[3]),
            _shorten(task, caps[4]), _shorten(row.split, caps[5]),
            _number(row.metrics["loss"]), _number(row.metrics["auroc"]),
            _number(row.metrics["aupr"]),
        ])
    widths = [
        min(caps[index], max(len(headers[index]), *(len(row[index]) for row in values)))
        for index in range(len(headers))
    ]

    def line(parts: list[str]) -> str:
        return "| " + " | ".join(
            _shorten(part, widths[index]).ljust(widths[index])
            for index, part in enumerate(parts)
        ) + " |"

    print(title)
    if not values:
        print("  (waiting for metrics)")
        return
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print(line(headers))
    print(separator)
    for row in values:
        print(line(row))
    print(separator)
    if len(rows) > max_rows:
        print(f"  ... {len(rows) - max_rows} more rows; use --max-rows to show them")


def render(monitor: Monitor, *, max_rows: int, clear: bool) -> None:
    if clear and sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training metrics monitor  {timestamp}")
    print(f"Sources: {len(monitor.specs)} configured, {len(monitor.readers)} readable")
    print()
    _render_table("TRAIN", monitor.store.table_rows("train"), max_rows)
    print()
    _render_table("VAL", monitor.store.table_rows("val"), max_rows)
    sys.stdout.flush()


def _parse_spec(raw: str) -> SourceSpec:
    if "=" not in raw:
        raise ValueError(f"input must use STAGE=PATH, got {raw!r}")
    stage, raw_path = raw.split("=", 1)
    if not raw_path:
        raise ValueError(f"input path is empty in {raw!r}")
    return SourceSpec(canonical_stage(stage), Path(raw_path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", default=[], metavar="STAGE=PATH",
        help="repeatable source specification; PATH may be a text log, run directory, or backup.swanlab",
    )
    parser.add_argument("--stage1-log", action="append", default=[], type=Path)
    parser.add_argument("--stage3-log", action="append", default=[], type=Path)
    parser.add_argument("--stage8-log", action="append", default=[], type=Path)
    parser.add_argument("--stage14-log", action="append", default=[], type=Path)
    parser.add_argument("--stage15-log", action="append", default=[], type=Path)
    parser.add_argument("--interval", type=float, default=2.0, help="polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="render one snapshot and exit")
    parser.add_argument("--no-clear", action="store_true", help="do not clear the terminal between snapshots")
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument(
        "--plot", action="store_true",
        help="open a Matplotlib window with live train/val curves",
    )
    parser.add_argument(
        "--plot-file", type=Path,
        help="also save/update the live Matplotlib figure at this path",
    )
    parser.add_argument(
        "--plot-stage", action="append", choices=tuple(STAGE_ORDER),
        help="restrict plotted panels to this stage; repeat for multiple stages",
    )
    parser.add_argument(
        "--plot-metric", action="append", choices=METRIC_NAMES,
        help="restrict plotted panels to this metric; repeat for multiple metrics",
    )
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.max_rows <= 0:
        parser.error("--max-rows must be positive")

    specs: list[SourceSpec] = []
    try:
        specs.extend(_parse_spec(raw) for raw in args.input)
        specs.extend(SourceSpec("stage1", path) for path in args.stage1_log)
        specs.extend(SourceSpec("stage3", path) for path in args.stage3_log)
        specs.extend(SourceSpec("stage8", path) for path in args.stage8_log)
        specs.extend(SourceSpec("stage14", path) for path in args.stage14_log)
        specs.extend(SourceSpec("stage15", path) for path in args.stage15_log)
    except ValueError as exc:
        parser.error(str(exc))
    if not specs:
        parser.error("provide at least one --input or --stage{1,3,8,14}-log")

    monitor = Monitor(specs)
    plotter: LivePlotter | None = None
    if args.plot or args.plot_file is not None:
        metric_titles = LivePlotter.PLOT_METRIC_TITLES
        selected_metrics = (
            tuple((metric, metric_titles[metric]) for metric in args.plot_metric)
            if args.plot_metric
            else None
        )
        try:
            plotter = LivePlotter(
                output=args.plot_file,
                show_window=args.plot,
                stages=args.plot_stage or None,
                metrics=selected_metrics,
            )
        except RuntimeError as exc:
            parser.error(str(exc))
    try:
        while True:
            monitor.poll(flush=args.once)
            render(monitor, max_rows=args.max_rows, clear=not args.no_clear and not args.once)
            if plotter is not None:
                plotter.update(monitor.store)
            if args.once:
                if plotter is not None:
                    plotter.show_once()
                break
            if plotter is not None:
                plotter.wait(args.interval)
            else:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
