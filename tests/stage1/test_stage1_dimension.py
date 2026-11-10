from pathlib import Path

from source.plm_unified.config import ModelConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_CONFIGS = (
    "plm_stage1.yaml",
    "plm_stage1_ph_pt.yaml",
    "plm_stage1_ph_pt_benchmark_test.yaml",
    "plm_stage1_tcr_eval.yaml",
)


def test_stage1_model_defaults_use_256_dimensions() -> None:
    assert ModelConfig().model_dim == 256


def test_stage1_configs_use_256_dimensions_and_matching_attention_width() -> None:
    for filename in STAGE1_CONFIGS:
        config = load_config(PROJECT_ROOT / "configs" / filename)

        assert config.model.model_dim == 256, filename
        assert config.model.num_attention_heads == 8, filename
        assert config.model.feedforward_dim == 1024, filename
