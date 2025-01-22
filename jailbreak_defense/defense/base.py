from dataclasses import dataclass, field

@dataclass
class DefenseConfig:
    defense_method: str  = field(default='BaseLine')
    display:        bool = field(default=False)

    def load_args(self, args):
        self.defense_method = args.defense_method
        self.display = args.display

class DefenseBase:
    def __init__(self, config, preloaded_model=None, **kwargs):
        self.defense_method = config.defense_method
        self.display = config.display
        self.preloaded_model = preloaded_model

    def defense(self, prompt, target_lm, response=None):
        raise NotImplementedError