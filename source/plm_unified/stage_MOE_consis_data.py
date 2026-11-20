"""Full triplet labels, projected-supervision audits and frozen residue caches."""
from __future__ import annotations

import csv
import hashlib
import json
from itertools import chain
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .cache import EmbeddingCache
from .encoders import checkpoint_fingerprint
from .stage_MOE_consis_model import TASKS

ENTITIES = {'ph': ('peptide', 'hla'), 'pt': ('peptide', 'tcr'), 'pht': ('peptide', 'hla', 'tcr')}
AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY')


class SupervisionAuditError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class StageMOEConsisRecord:
    peptide: str
    hla: str | None
    tcr: str | None
    labels: tuple[float, float, float]
    masks: tuple[bool, bool, bool]
    source_file: str
    source_row: int
    source: str = ''
    split_type: str = ''
    class_code: str = ''

    def key(self, task):
        return tuple(getattr(self, entity) for entity in ENTITIES[task])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_task_records(path, task, model, *, expected_split=None, ph_only_triplets=False):
    """Read native labels and the triplet supervision used by the experiment."""
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}')
    path = Path(path)
    records, seen = [], {}
    stats = {'input_rows': 0, 'duplicates': 0, 'retained_rows': 0, 'positive_rows': 0,
             'invalid_sequence_rows': 0, 'rejected_rows': [],
             'source_file': str(path), 'sha256': file_sha256(path), 'label_policy': 'triplet_ph_supervision' if ph_only_triplets else 'triplet_full_supervision'}
    label_column = 'pht' if task == 'pht' else 'label'
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = [name.strip().lower() for name in (reader.fieldnames or [])]
        if len(set(columns)) != len(columns):
            raise ValueError(f'Duplicate normalized CSV columns: {path}')
        required = set(ENTITIES[task]) | {label_column}
        if task == 'pht' and expected_split in ('train', 'val'):
            if not ph_only_triplets:
                required.update(('ph', 'pt', 'class_code'))
            elif expected_split == 'train':
                required.add('ph')
        if not required.issubset(columns):
            raise ValueError(f'{path}: missing columns {sorted(required - set(columns))}')
        for rownum, raw in enumerate(reader, 2):
            stats['input_rows'] += 1
            row = {k.strip().lower(): (v or '').strip() for k, v in raw.items() if k is not None}
            split = row.get('split_type', '')
            if expected_split and split and not (split == expected_split or split.startswith(expected_split + '_')):
                raise ValueError(f'{path}:{rownum}: wrong split_type {split!r}')
            entities = dict(peptide=None, hla=None, tcr=None)
            invalid = []
            for entity in ENTITIES[task]:
                sequence = ''.join(row[entity].split()).upper()
                minimum = model.peptide_min_length if entity == 'peptide' else 1
                maximum = getattr(model, f'{entity}_max_length')
                if not minimum <= len(sequence) <= maximum or not set(sequence).issubset(AMINO_ACIDS):
                    invalid.append({'entity': entity, 'sequence': sequence,
                                    'reason': 'length_or_noncanonical_amino_acids'})
                entities[entity] = sequence
            if invalid:
                stats['invalid_sequence_rows'] += 1
                stats['rejected_rows'].append({'source_row': rownum, 'invalid': invalid})
                continue
            try:
                label = float(row[label_column])
            except ValueError as exc:
                raise ValueError(f'{path}:{rownum}: invalid native label') from exc
            if label not in (0.0, 1.0):
                raise ValueError(f'{path}:{rownum}: native label must be 0 or 1')
            labels, masks = [float('nan')] * 3, [False] * 3
            labels[TASKS.index(task)], masks[TASKS.index(task)] = label, True
            if task == 'pht':
                code = row.get('class_code', '')
                if code and code not in ('000', '010', '100', '111'):
                    raise ValueError(f'{path}:{rownum}: invalid class_code {code!r}')
                auxiliary_tasks = ('ph',) if ph_only_triplets else ('ph', 'pt')
                for index, name in enumerate(auxiliary_tasks):
                    raw_label = row.get(name, '')
                    if not raw_label:
                        if expected_split == 'train' or (expected_split == 'val' and not ph_only_triplets):
                            raise ValueError(f'{path}:{rownum}: missing {name} label')
                        continue
                    value = float(raw_label)
                    if value not in (0.0, 1.0):
                        raise ValueError(f'{path}:{rownum}: invalid {name} label')
                    labels[index], masks[index] = value, True
                if not ph_only_triplets and expected_split in ('train', 'val') and not code:
                    raise ValueError(f'{path}:{rownum}: missing class_code')
                if code and any(masks[i] and labels[i] != int(code[i]) for i in range(3)):
                    raise ValueError(f'{path}:{rownum}: class_code/labels mismatch')
                if ph_only_triplets and masks[0] and labels[0] == 0 and label == 1:
                    raise ValueError(f'{path}:{rownum}: PHT-positive triplet requires a positive PH label')
                if all(masks) and ''.join(str(int(v)) for v in labels) not in ('000', '010', '100', '111'):
                    raise ValueError(f'{path}:{rownum}: invalid triplet labels')
            record = StageMOEConsisRecord(**entities, labels=tuple(labels), masks=tuple(masks),
                                    source_file=str(path), source_row=rownum,
                                    source=row.get('source', row.get('source_file', 'unknown')),
                                    split_type=split, class_code=row.get('class_code', '') if task == 'pht' else '')
            key = record.key(task)
            signature = tuple(v if valid else None for v, valid in zip(labels, masks))
            if key in seen:
                if seen[key] != signature:
                    raise ValueError(f'{path}:{rownum}: conflicting {task} labels for {key}')
                stats['duplicates'] += 1
                continue
            seen[key] = signature
            records.append(record)
            stats['positive_rows'] += int(label)
    if not records:
        raise ValueError(f'No records found: {path}')
    stats['retained_rows'] = len(records)
    stats['negative_rows'] = len(records) - stats['positive_rows']
    return records, stats


def audit_stage_MOE_consis_data(config, splits=('train', 'val')):
    ph_only = getattr(config, 'ph_only_triplets', False)
    policy = 'triplet_ph_supervision' if ph_only else 'triplet_full_supervision'
    datasets, report = {}, {'label_policy': policy, 'files': {}, 'entity_overlap': {}}
    digest = hashlib.sha256(f'stage_MOE_consis_{policy}_v1'.encode())
    for split in splits:
        datasets[split] = {}
        for task in TASKS:
            path = config.paths.data_root / task.upper() / f'{split}.csv'
            records, stats = read_task_records(path, task, config.model, expected_split=split, ph_only_triplets=ph_only)
            if split in ('train', 'val') and not (stats['positive_rows'] and stats['negative_rows']):
                raise ValueError(f'{path}: training/validation require both classes')
            datasets[split][task] = records
            report['files'][f'{task}/{split}'] = stats
            digest.update(f'{task}/{split}:{stats["sha256"]}'.encode())
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            for task in TASKS:
                overlap = {r.key(task) for r in datasets[left][task]} & {r.key(task) for r in datasets[right][task]}
                if overlap:
                    raise ValueError(f'{task}: {len(overlap)} supervised records overlap {left}/{right}')
            report['entity_overlap'][f'{left}/{right}'] = {}
            for entity in ('peptide', 'hla', 'tcr'):
                def values(split):
                    return {getattr(r, entity) for rows in datasets[split].values() for r in rows if getattr(r, entity)}
                report['entity_overlap'][f'{left}/{right}'][entity] = len(values(left) & values(right))
    report['cold_interpretation'] = 'Aggregate splits may mix scenarios. Strict cold eligibility is checked per benchmark against all task training pools.'
    report['dataset_fingerprint'] = digest.hexdigest()
    report['binary_supervision_audit'] = audit_binary_supervision(config, datasets)
    return datasets, report


def audit_binary_supervision(config, datasets):
    """Audit projected supervision, without changing rows or split membership."""
    summary = {}
    train = datasets.get('train', {})
    for index, task in enumerate(('ph', 'pt')):
        known = {}
        for split, tasks in datasets.items():
            supervision = {}
            for record in chain(tasks.get(task, ()), tasks.get('pht', ())):
                if not record.masks[index]:
                    continue
                key, label = record.key(task), record.labels[index]
                previous = supervision.get(key)
                location = f'{record.source_file}:{record.source_row}'
                if previous is not None and previous[0] != label:
                    raise ValueError(f'{task} binary supervision conflict: {previous[1]} vs {location}')
                supervision[key] = (label, location)
            if split == 'train':
                known = supervision
        # Native heldout pair tasks must not receive triplet-derived training labels.
        checked = 0
        leaks, examples = 0, []
        if train:
            paths = [config.paths.data_root / task.upper() / f'{split}.csv' for split in ('val', 'test')]
            paths += sorted((config.paths.data_root / task.upper() / 'benchmarks').glob('test_*.csv'))
            for path in paths:
                if not path.exists():
                    continue
                split = 'val' if path.stem == 'val' else 'test'
                rows, _ = read_task_records(path, task, config.model, expected_split=split)
                for record in rows:
                    if record.key(task) in known:
                        leaks += 1
                        if len(examples) < 20:
                            examples.append({'pair': record.key(task),
                                             'training': known[record.key(task)][1],
                                             'heldout': f'{path}:{record.source_row}'})
                checked += len(rows)
        summary[task] = {'training_pairs': len(known), 'heldout_rows_checked': checked,
                         'overlap_rows': leaks, 'overlap_examples': examples}
    if any(value['overlap_rows'] for value in summary.values()):
        counts = ', '.join(f'{task}={value["overlap_rows"]}' for task, value in summary.items())
        example = next(value['overlap_examples'][0] for value in summary.values() if value['overlap_rows'])
        raise SupervisionAuditError(f'binary supervision leakage ({counts} heldout rows): '
                                    f'{example["training"]} vs {example["heldout"]}', summary)
    return summary


def collect_sequences(datasets):
    sequences = {e: set() for e in ('peptide', 'hla', 'tcr')}
    for tasks in datasets.values():
        for rows in tasks.values():
            for record in rows:
                for entity in sequences:
                    if getattr(record, entity):
                        sequences[entity].add(getattr(record, entity))
    return sequences


def open_caches(config):
    fingerprints = {entity: checkpoint_fingerprint(getattr(config.paths, 'tcr_bert_model' if entity == 'tcr' else 'esmc_model'))
                    for entity in ('peptide', 'hla', 'tcr')}
    caches = {}
    try:
        for entity, fingerprint in fingerprints.items():
            cache = EmbeddingCache(config.paths.cache_root / entity,
                                   expected_hidden_dim=getattr(config.model, f'{entity}_input_dim'),
                                   expected_fingerprint=fingerprint)
            caches[entity] = cache
            if cache.max_length != getattr(config.model, f'{entity}_max_length'):
                raise ValueError(f'{entity}: cache maximum length mismatch')
            manifest = cache.manifest
            if manifest.get('entity') != entity or manifest.get('dtype') != 'float16':
                raise ValueError(f'{entity}: cache entity/dtype mismatch')
    except Exception:
        for cache in caches.values():
            cache.close()
        raise
    return caches


def cache_identity(caches):
    return {entity: hashlib.sha256(json.dumps(cache.manifest, sort_keys=True).encode()).hexdigest()
            for entity, cache in caches.items()}


class CachedStageMOEConsisDataset(Dataset):
    def __init__(self, records, task, caches):
        self.records, self.task = records, task
        self.locations = [{entity: caches[entity].resolve(getattr(r, entity)) for entity in ENTITIES[task]} for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index], self.locations[index]


class StageMOEConsisCollator:
    def __init__(self, caches, task):
        self.readers = {e: caches[e].worker_reader() for e in ENTITIES[task]}

    def __call__(self, rows):
        records, locations = zip(*rows)
        batch = {'labels': {}, 'masks': {}, 'records': records}
        for index, task in enumerate(TASKS):
            batch['labels'][task] = torch.tensor([r.labels[index] for r in records], dtype=torch.float32)
            batch['masks'][task] = torch.tensor([r.masks[index] for r in records], dtype=torch.bool)
        for entity, reader in self.readers.items():
            hidden, mask = reader.get_many([loc[entity] for loc in locations])
            batch[f'{entity}_hidden'] = torch.from_numpy(hidden)
            batch[f'{entity}_mask'] = torch.from_numpy(mask)
        return batch


def limited_records(records, limit, seed):
    if limit is None or len(records) <= limit:
        return records
    indices = np.random.default_rng(seed).permutation(len(records))[:limit]
    return [records[i] for i in indices]


def make_loader(records, task, caches, settings, *, training=False, epoch=0):
    limit = settings.limit_train_samples if training else settings.limit_eval_samples
    records = limited_records(records, limit, settings.seed)
    dataset = CachedStageMOEConsisDataset(records, task, caches)
    generator = torch.Generator().manual_seed(settings.seed + 1009 * epoch + TASKS.index(task))
    return DataLoader(dataset, batch_size=settings.batch_size if training else settings.eval_batch_size,
                      shuffle=training, drop_last=False, num_workers=settings.num_workers,
                      collate_fn=StageMOEConsisCollator(caches, task), generator=generator)


def move_batch(batch, device):
    return {key: ({t: value.to(device) for t, value in val.items()} if key in ('labels', 'masks')
                  else val.to(device=device, dtype=torch.float32 if key.endswith('_hidden') else val.dtype)
                  if isinstance(val, torch.Tensor) else val) for key, val in batch.items()}
