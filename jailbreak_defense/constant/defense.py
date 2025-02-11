from ..defense import *
from .defenseconfig import DefenseConfig_Default, BacktranslationConfig_Default

DEFENSE_DICT = {
    'None': DefenseBase,
     None : DefenseBase,
    'backtranslation': BacktranslationDefense
}

DEFENSE_CONFIG_DICT = {
    'None': DefenseConfig_Default,
     None : DefenseConfig_Default,
    'backtranslation': BacktranslationConfig_Default
}

