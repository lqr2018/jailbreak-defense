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