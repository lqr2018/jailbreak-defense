class DefenseBase:
    def __init__(self, config, preloaded_model=None, **kwargs):
        self.defense_method = config.defense_method
        self.display = config.display
        self.preloaded_model = preloaded_model

    def defense(self, prompt, target_lm, response=None):
        raise NotImplementedError