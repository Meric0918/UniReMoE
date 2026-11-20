"""Run the sample-gated Consis stage with the shared PT-fusion lifecycle."""
import sys

from plm_unified.stage_final_cli import main
from plm_unified.stage_MOE_consis_PT_config import StageMOEConsisPTConfig, load_stage_MOE_consis_PT_config
from plm_unified.stage_MOE_consis_PT_evaluation import evaluation_setup
from plm_unified.stage_MOE_consis_PT_model import StageMOEConsisPTBindingModel


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:], stage=StageMOEConsisPTConfig.stage_name,
         config_loader=load_stage_MOE_consis_PT_config, model_type=StageMOEConsisPTBindingModel,
         evaluation_setup=evaluation_setup)
