"""Cached-feature teacher and student binding models for Stage 10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .config import ModelConfig
from .triplet_model import TripletBindingModel


@dataclass
class Stage10StudentOutput:
    phla_logits: torch.Tensor
    ptcr_logits: torch.Tensor
    pht_logits: torch.Tensor


@dataclass
class Stage10TeacherOutput:
    phla_logits: torch.Tensor
    ptcr_logits: torch.Tensor


class Stage10BindingModel(TripletBindingModel):
    """Stage-3 downstream modules operating on frozen residue features."""

    def _forward_legacy_representations(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
        tcr_hidden: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        peptide = self.peptide_adapter(peptide_hidden, peptide_mask)
        hla = self.hla_adapter(hla_hidden, hla_mask)
        tcr = self.tcr_adapter(tcr_hidden, tcr_mask)
        return peptide, hla, tcr

    def forward_legacy_heads(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
        tcr_hidden: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> Stage10TeacherOutput:
        peptide, hla, tcr = self._forward_legacy_representations(
            peptide_hidden,
            peptide_mask,
            hla_hidden,
            hla_mask,
            tcr_hidden,
            tcr_mask,
        )
        phla_sequence, _ = self.phla_cross_attention(
            peptide,
            hla,
            peptide_mask,
            hla_mask,
            return_attention=False,
        )
        phla_pooled, _ = self.phla_pooling(
            phla_sequence,
            peptide_mask,
            return_weights=False,
        )
        ptcr_sequence, _ = self.ptcr_cross_attention(
            peptide,
            tcr,
            peptide_mask,
            tcr_mask,
            return_attention=False,
        )
        ptcr_pooled, _ = self.ptcr_pooling(
            ptcr_sequence,
            peptide_mask,
            return_weights=False,
        )
        return Stage10TeacherOutput(
            phla_logits=self.phla_classifier(phla_pooled),
            ptcr_logits=self.ptcr_classifier(ptcr_pooled),
        )

    def forward_all(
        self,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_hidden: torch.Tensor,
        hla_mask: torch.Tensor,
        tcr_hidden: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> Stage10StudentOutput:
        peptide, hla, tcr = self._forward_legacy_representations(
            peptide_hidden,
            peptide_mask,
            hla_hidden,
            hla_mask,
            tcr_hidden,
            tcr_mask,
        )
        phla_sequence, _ = self.phla_cross_attention(
            peptide,
            hla,
            peptide_mask,
            hla_mask,
            return_attention=False,
        )
        phla_pooled, _ = self.phla_pooling(
            phla_sequence,
            peptide_mask,
            return_weights=False,
        )
        ptcr_sequence, _ = self.ptcr_cross_attention(
            peptide,
            tcr,
            peptide_mask,
            tcr_mask,
            return_attention=False,
        )
        ptcr_pooled, _ = self.ptcr_pooling(
            ptcr_sequence,
            peptide_mask,
            return_weights=False,
        )
        pht_sequence, _ = self.triplet_cross_attention(
            phla_sequence,
            tcr,
            peptide_mask,
            tcr_mask,
            return_attention=False,
        )
        pht_pooled, _ = self.triplet_pooling(
            pht_sequence,
            peptide_mask,
            return_weights=False,
        )
        return Stage10StudentOutput(
            phla_logits=self.phla_classifier(phla_pooled),
            ptcr_logits=self.ptcr_classifier(ptcr_pooled),
            pht_logits=self.triplet_classifier(pht_pooled),
        )


class FrozenStage10Teacher(Stage10BindingModel):
    """Binding model that cannot be switched into training mode."""

    def train(self, mode: bool = True):
        return super().train(False)


@dataclass
class Stage10TeacherStudentModels:
    teacher: FrozenStage10Teacher
    student: Stage10BindingModel

    def assert_teacher_frozen(self) -> None:
        if self.teacher.training:
            raise RuntimeError(
                "Stage-10 teacher must remain in evaluation mode"
            )
        if any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in self.teacher.parameters()
        ):
            raise RuntimeError(
                "Stage-10 teacher must remain frozen and gradient-free"
            )


def build_stage10_teacher_student_models(
    config: ModelConfig,
    stage3_binding_state: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> Stage10TeacherStudentModels:
    teacher = FrozenStage10Teacher(config)
    student = Stage10BindingModel(config)
    teacher.load_state_dict(stage3_binding_state, strict=True)
    student.load_state_dict(stage3_binding_state, strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    student.requires_grad_(True)
    models = Stage10TeacherStudentModels(
        teacher=teacher.to(device),
        student=student.to(device),
    )
    models.assert_teacher_frozen()
    return models
