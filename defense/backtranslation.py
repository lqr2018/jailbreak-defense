import math
from dataclasses import dataclass, field
from .base import DefenseBase, DefenseConfig
from judge import check_rejection
from model import TargetLM


@dataclass
class BacktranslationConfig(DefenseConfig):
    threshold: float = field(default=-2.0)
    infer_model: str = field(default="vicuna")
    infer_lm_length: int = field(default=None)
    new_response_length: int = field(default=None)
    defense_lm_max_memory: float = field(default=None)
    return_new_response_anyway: bool = field(default=False)

    def __post_init__(self):
        self.defense_method = "backtranslation"

    def load_args(self, args):
        # TODO need to be simplified
        super().load_args(args)
        self.threshold = args.backtranslation_threshold
        self.infer_model = args.backtranslation_infer_model
        self.infer_lm_length = args.target_max_new_tokens
        self.new_response_length = args.backtranslation_new_response_length
        self.defense_lm_max_memory = args.max_memory
        self.return_new_response_anyway = args.return_new_response_anyway


class BacktranslationDefense(DefenseBase):
    def __init__(self, config, preloaded_model, **kwargs):
        super().__init__(config)
        print(f"using threshold {config.threshold} for backtranslation")
        infer_lm = TargetLM(model_name=config.infer_model,
                            max_new_tokens=config.infer_lm_length,
                            max_memory=config.defense_lm_max_memory,
                            preloaded_model=preloaded_model)
        self.infer_lm = infer_lm
        self.threshold = config.threshold
        self.return_new_response_anyway = config.return_new_response_anyway
        self.new_response_length = config.new_response_length

    def defense(self, prompt, target_lm, response=None):
        if self.verbose:
            print("adversarial harmful prompt: \n", prompt)
        if response is None:
            response = target_lm.get_response(prompt, verbose=self.verbose)

        if check_rejection()