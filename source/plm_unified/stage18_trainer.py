"""Stage18 replay/KD policies share exact same trajectory/selection machinery."""

from .stage16_trainer import Stage16Trainer


class Stage18Trainer(Stage16Trainer):
    def __init__(self, model, config, data, **kwargs):
        if config.stage != 18:
            raise ValueError("Stage18Trainer requires stage=18")
        super().__init__(model, config, data, **kwargs)
