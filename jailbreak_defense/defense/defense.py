from .backtranslation import BacktranslationDefense, BacktranslationConfig
from .base import DefenseBase, DefenseConfig


DEFENSE_DICT = {
    'None': DefenseBase,
     None : DefenseBase,
    'backtranslation': BacktranslationDefense
}

DEFENSE_CONFIG_DICT = {
    'None': DefenseConfig,
     None : DefenseConfig,
    'backtranslation': BacktranslationConfig
}


def args_to_defense_config(args):
    defense_method = args.defense_method
    if defense_method.startswith("backtranslation"):
        defense_method = "backtranslation"
    config = DEFENSE_CONFIG_DICT[defense_method](args.defense_method)
    config.load_args(args)
    return config


def load_defense(defense_config, preload_model=None):
    defense_method = defense_config.defense_method
    if defense_method.startswith("backtranslation"):
        if 'threshold' in defense_method:
            threshold = float(defense_method.split('_')[-1])
            defense_method = "backtranslation"
            if threshold > 0:
                threshold = -threshold
            defense_config.backtranslation_threshold = threshold
        else:
            defense_config.backtranslation_threshold = -2.0

    return DEFENSE_DICT[defense_method](defense_config, preload_model)