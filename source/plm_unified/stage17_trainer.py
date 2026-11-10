"""Stage17 uses the common controlled trainer with tested freeze policies."""

from .stage16_trainer import Stage16Trainer


class Stage17Trainer(Stage16Trainer):
    def __init__(self, model, config, data, **kwargs):
        if config.stage != 17:
            raise ValueError("Stage17Trainer requires stage=17")
        super().__init__(model, config, data, **kwargs)
