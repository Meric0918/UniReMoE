"""Run the PH–PHT consistency experiment."""
import sys

from plm_unified.stage_final_cli import main
from plm_unified.stage_final_new import StageFinalNewBindingModel, StageFinalNewConfig, load_stage_final_new_config


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:], stage=StageFinalNewConfig.stage_name,
         config_loader=load_stage_final_new_config, model_type=StageFinalNewBindingModel)
