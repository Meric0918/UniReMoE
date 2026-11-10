"""PT residual fusion with all-head consistency supervision."""
from .stage_MOE_PT_model import StageMOEPTBindingModel, StageMOEPTOutput as StageFinalOutput
from .stage_MOE_model import TASKS
from .stage_MOE_consis_model import forward_batch

ARCHITECTURE = 'stage_final_v1'


class StageFinalBindingModel(StageMOEPTBindingModel):
    architecture = ARCHITECTURE

    def __init__(self, config):
        super().__init__(config)
        self.consistency_margin = config.stage_final.consistency_margin
