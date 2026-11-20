"""PT residual fusion with all-head consistency supervision."""
from .stage_MOE_PT_model import StageMOEPTBindingModel, StageMOEPTOutput as StageFinalOutput
from .stage_final_loss import compute_stage_final_loss
from .stage_MOE_model import TASKS
from .stage_MOE_consis_model import forward_batch

ARCHITECTURE = 'stage_final_v1'


class StageFinalBindingModel(StageMOEPTBindingModel):
    architecture = ARCHITECTURE
    compute_loss = staticmethod(compute_stage_final_loss)

    def __init__(self, config):
        super().__init__(config)
        self.ph_only_triplets = config.ph_only_triplets
        self.consistency_margin = config.stage_final.consistency_margin
