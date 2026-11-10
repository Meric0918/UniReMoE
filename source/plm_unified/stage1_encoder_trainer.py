"""Sequence-encoder variant of the shared Stage1 training lifecycle."""

from .trainer import Stage1Trainer, build_optimizer


def build_encoder_optimizer(model, config):
    training = config.training

    def rate(name, fallback):
        value = getattr(training, name)
        return fallback if value is None else value

    specs = [
        ("shared_peptide_encoder", model.peptide_encoder,
         rate("shared_peptide_encoder_lr", training.encoder_lr)),
        ("hla_encoder", model.hla_encoder, rate("hla_encoder_lr", training.encoder_lr)),
        ("tcr_encoder", model.tcr_encoder, rate("tcr_encoder_lr", training.encoder_lr)),
    ]
    for task in ("phla", "ptcr"):
        for suffix in ("cross_attention", "pooling", "classifier"):
            name = f"{task}_{suffix}"
            specs.append((name, getattr(model, name), rate(
                f"{name}_lr", getattr(training, f"{suffix}_lr"),
            )))
    return build_optimizer(model, config, module_specs=tuple(specs))


class Stage1EncoderTrainer(Stage1Trainer):
    display_name = "Stage 1 Encoder"
    artifact_namespace = "stage1_encoder"
    checkpoint_architecture = "stage1_encoder"
    shared_freeze_state_key = "freeze_shared_encoder"
    joint_checkpoint_type = "stage1_encoder_joint_model"

    def _build_optimizer(self, model, config):
        return build_encoder_optimizer(model, config)

    def _feature_embeddings(self, output):
        return {
            "plm": output.plm_embedding,
            "sequence_encoder": output.sequence_encoder_embedding,
            "classifier_input": output.pooled_embedding,
        }
