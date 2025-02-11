from dataclasses import dataclass, field


@dataclass
class DefenseConfig:
    defense_method: str  = field(default='BaseLine')
    display:        bool = field(default=False)

    def load_args(self, args):
        self.defense_method = args.defense_method
        self.display = args.display


@dataclass
class BacktranslationConfig(DefenseConfig):
    threshold: float = field(default=-2.0)
    infer_model: str = field(default="llama-3-8b")
    infer_lm_length: int = field(default=None)
    new_response_length: int = field(default=None)
    defense_lm_max_memory: float = field(default=None)
    return_new_response_anyway: bool = field(default=False)

    def __post_init__(self):
        self.defense_method = "backtranslation"

    def load_args(self, args):
        super().load_args(args)
        self.threshold = args.backtranslation_threshold
        self.infer_model = args.backtranslation_infer_model
        self.infer_lm_length = args.target_max_new_tokens
        self.new_response_length = args.backtranslation_new_response_length
        self.defense_lm_max_memory = args.max_memory
        self.return_new_response_anyway = args.return_new_response_anyway

DefenseConfig_Default = DefenseConfig()
BacktranslationConfig_Default = BacktranslationConfig()