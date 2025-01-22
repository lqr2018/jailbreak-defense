from dataclasses import dataclass, field

@dataclass
class DefenseConfig:
    defense_method: str  = field(default='BaseLine')
    verbose:        bool = field(default=False)

    def load_args(self, args):
        self.defense_method = args.defense_method
        self.verbose = args.verbose

class DefenseBase:
    def __init__(self, config, preloaded_model=None, **kwargs):
        self.defense_method = config.defense_method
        self.verbose = config.verbose
        self.preloaded_model = preloaded_model

    def defense(self, prompt, target_lm, response=None):
        raise NotImplementedError