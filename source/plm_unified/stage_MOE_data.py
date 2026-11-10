"""Native task labels, split auditing and frozen residue-cache loaders."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .cache import EmbeddingCache
from .encoders import checkpoint_fingerprint
from .stage_MOE_model import TASKS

ENTITIES = {'ph': ('peptide', 'hla'), 'pt': ('peptide', 'tcr'), 'pht': ('peptide', 'hla', 'tcr')}
AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY')


@dataclass(frozen=True)
class StageMOERecord:
    peptide: str
    hla: str | None
    tcr: str | None
    labels: tuple[float, float, float]
    masks: tuple[bool, bool, bool]
    source_file: str
    source_row: int
    source: str = ''
    split_type: str = ''

    def key(self, task):
        return tuple(getattr(self, entity) for entity in ENTITIES[task])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_task_records(path, task, model, *, expected_split=None):
    """Only the file's native task supplies supervision; other label columns are ignored."""
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}')
    path = Path(path)
    records, seen = [], {}
    stats = {'input_rows': 0, 'duplicates': 0, 'retained_rows': 0, 'positive_rows': 0,
             'invalid_sequence_rows': 0, 'rejected_rows': [],
             'source_file': str(path), 'sha256': file_sha256(path), 'label_policy': 'native_task_only'}
    label_column = 'pht' if task == 'pht' else 'label'
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = [name.strip().lower() for name in (reader.fieldnames or [])]
        if len(set(columns)) != len(columns):
            raise ValueError(f'Duplicate normalized CSV columns: {path}')
        required = set(ENTITIES[task]) | {label_column}
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
            record = StageMOERecord(**entities, labels=tuple(labels), masks=tuple(masks),
                                    source_file=str(path), source_row=rownum,
                                    source=row.get('source', row.get('source_file', 'unknown')),
                                    split_type=split)
            key = record.key(task)
            if key in seen:
                if seen[key] != label:
                    raise ValueError(f'{path}:{rownum}: conflicting {task} labels for {key}')
                stats['duplicates'] += 1
                continue
            seen[key] = label
            records.append(record)
            stats['positive_rows'] += int(label)
    if not records:
        raise ValueError(f'No records found: {path}')
    stats['retained_rows'] = len(records)
    stats['negative_rows'] = len(records) - stats['positive_rows']
    return records, stats


def audit_stage_MOE_data(config, splits=('train', 'val')):
    datasets, report = {}, {'label_policy': 'native_task_only', 'files': {}, 'entity_overlap': {}}
    digest = hashlib.sha256(b'stage_MOE_native_task_only_v1')
    for split in splits:
        datasets[split] = {}
        for task in TASKS:
            path = config.paths.data_root / task.upper() / f'{split}.csv'
            records, stats = read_task_records(path, task, config.model, expected_split=split)
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
    return datasets, report


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


class CachedStageMOEDataset(Dataset):
    def __init__(self, records, task, caches):
        self.records, self.task = records, task
        self.locations = [{entity: caches[entity].resolve(getattr(r, entity)) for entity in ENTITIES[task]} for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index], self.locations[index]


class StageMOECollator:
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
    dataset = CachedStageMOEDataset(records, task, caches)
    generator = torch.Generator().manual_seed(settings.seed + 1009 * epoch + TASKS.index(task))
    return DataLoader(dataset, batch_size=settings.batch_size if training else settings.eval_batch_size,
                      shuffle=training, drop_last=False, num_workers=settings.num_workers,
                      collate_fn=StageMOECollator(caches, task), generator=generator)


def move_batch(batch, device):
    return {key: ({t: value.to(device) for t, value in val.items()} if key in ('labels', 'masks')
                  else val.to(device=device, dtype=torch.float32 if key.endswith('_hidden') else val.dtype)
                  if isinstance(val, torch.Tensor) else val) for key, val in batch.items()}
