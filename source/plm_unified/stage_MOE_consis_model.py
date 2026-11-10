"""Stage MoE architecture with an independent experiment identity."""
from .stage_MOE_model import StageMOEBindingModel, TASKS, StageMOEOutput

ARCHITECTURE = 'stage_MOE_consis_v1'


class StageMOEConsisBindingModel(StageMOEBindingModel):
    architecture = ARCHITECTURE

    def __init__(self, config):
        super().__init__(config)
        self.consistency_margin = config.stage_MOE_consis.consistency_margin


def forward_batch(model, task, batch):
    p = (batch['peptide_hidden'], batch['peptide_mask'])
    if task == 'pht':
        return model.forward_all(*p, batch['hla_hidden'], batch['hla_mask'],
                                 batch['tcr_hidden'], batch['tcr_mask'])
    entity = 'hla' if task == 'ph' else 'tcr'
    return model(task, *p, batch[f'{entity}_hidden'], batch[f'{entity}_mask'])
