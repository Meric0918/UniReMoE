#!/usr/bin/env python3
"""Precompute frozen Stage11 residue caches; train and test scopes stay separate."""

import gc
import json

import torch

from plm_unified.cache import EmbeddingCacheWriter, batched
from plm_unified.encoders import ESMCResidueEncoder, TCRBertResidueEncoder, encoder_metadata
from plm_unified.stage11_cli import common_parser, config_from_args
from plm_unified.stage11_data import prepare_stage11_data, read_stage11_test, sequences_for_records, test_paths
from plm_unified.trainer import resolve_device


def write_entity_cache(encoder, sequences, config, entity, *, scope, overwrite=False):
    with EmbeddingCacheWriter(
        config.cache_dir(scope), entity, hidden_dim=encoder.hidden_size, max_length=encoder.max_length,
        model_metadata=encoder_metadata(encoder), dtype=config.cache.dtype,
        shard_size=config.cache.shard_size, chunk_rows=1, compression=config.cache.compression,
        overwrite=overwrite,
    ) as writer:
        for index, batch in enumerate(batched(sorted(sequences), config.cache.precompute_batch_size), 1):
            with torch.inference_mode():
                features = encoder.encode(batch)
            writer.add_batch(batch, features.hidden.cpu().to(torch.float16).numpy(),
                             features.mask.cpu().numpy(), features.lengths.cpu().numpy())
            if index == 1 or index % 100 == 0:
                print(f"stage11 {scope}/{entity}: {min(index * config.cache.precompute_batch_size, len(sequences))}/{len(sequences)}", flush=True)


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--scope", choices=("train", "test"), default="train")
    parser.add_argument("--entities", nargs="+", choices=("peptide", "hla", "tcr"), default=("peptide", "hla", "tcr"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    config = config_from_args(args)
    destination = config.cache_dir(args.scope)
    if args.scope == "train":
        data = prepare_stage11_data(config)
        sequences = data.sequences()
        report = {"scope": "train", "fingerprint": data.fingerprint, "audit": data.report}
    else:
        records, reports = [], {}
        for name, path in test_paths(config).items():
            values, reports[name] = read_stage11_test(path, config)
            records.extend(values)
        sequences = sequences_for_records(records)
        report = {"scope": "test", "audit": reports}
    report["unique_sequences"] = {key: len(value) for key, value in sequences.items()}
    if args.audit_only:
        print(json.dumps({"scope": args.scope, "unique_sequences": report["unique_sequences"]}, indent=2))
        return report
    marker = destination / "stage11_cache.json"
    if destination.exists() and any(destination.iterdir()) and not marker.exists():
        raise ValueError("Refusing to modify a non-Stage11 cache directory")
    selected = set(args.entities)
    for entity in selected:
        if (destination / entity).exists() and not args.overwrite:
            raise FileExistsError(f"{destination / entity} exists; select missing entities or explicitly use --overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"architecture": "stage11", "scope": args.scope}, indent=2))
    device = resolve_device(config.runtime.device)
    if selected & {"peptide", "hla"}:
        encoder = ESMCResidueEncoder(config.paths.esmc_model, max_length=config.model.hla_max_length,
                                     device=device, dtype=config.cache.dtype)
        try:
            for entity in ("peptide", "hla"):
                if entity in selected:
                    if encoder.hidden_size != getattr(config.model, f"{entity}_input_dim"):
                        raise ValueError("ESM-C output dimension differs from Stage11 configuration")
                    encoder.max_length = getattr(config.model, f"{entity}_max_length")
                    write_entity_cache(encoder, sequences[entity], config, entity, scope=args.scope, overwrite=args.overwrite)
        finally:
            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if "tcr" in selected:
        encoder = TCRBertResidueEncoder(config.paths.tcr_bert_model, max_length=config.model.tcr_max_length,
                                       device=device, dtype=config.cache.dtype)
        if encoder.hidden_size != config.model.tcr_input_dim:
            raise ValueError("TCR-BERT output dimension differs from Stage11 configuration")
        write_entity_cache(encoder, sequences["tcr"], config, "tcr", scope=args.scope, overwrite=args.overwrite)
    (destination / "sequence_coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["unique_sequences"], indent=2))
    return report


if __name__ == "__main__":
    main()
