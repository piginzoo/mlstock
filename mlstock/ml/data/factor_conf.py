from mlstock.factors.alpha_beta import AlphaBeta
from mlstock.factors.daily_indicator import DailyIndicator
from mlstock.factors.ff3_residual_std import FF3ResidualStd
from mlstock.factors.finance_indicator import FinanceIndicator
from mlstock.factors.kdj import KDJ
from mlstock.factors.macd import MACD
from mlstock.factors.psy import PSY
from mlstock.factors.rsi import RSI
from mlstock.factors.balance_sheet import BalanceSheet
from mlstock.factors.cashflow import CashFlow
from mlstock.factors.income import Income
from mlstock.factors.stake_holder import StakeHolder
from mlstock.factors.std import Std
from mlstock.factors.returns import Return
from mlstock.factors.turnover_return import TurnoverReturn

"""
所有的因子配置，有的因子类只包含一个feature，有的因子类可能包含多个features。
"""

# 测试用
# FACTORS = [TurnoverReturn,
#            Return,
#            Std,
#            MACD,
#            KDJ,
#            PSY,
#            RSI]

# 正式
FACTORS = [TurnoverReturn,
           Return,
           Std,
           MACD,
           KDJ,
           PSY,
           RSI,
           BalanceSheet,
           Income,
           CashFlow,
           FinanceIndicator,
           DailyIndicator,
           # FF3ResidualStd, # 暂时不用，太慢了，单独跑效果也一般
           AlphaBeta,
           StakeHolder]


def get_factor_names():
    """
    获得所有的因子名
    :return:
    """

    names = []
    for f in FACTORS:
        _names = f(None, None).name
        if type(_names) == list:
            names += _names
        else:
            names += [_names]
    return names


def get_factor_class_by_name(name):
    """
    根据名字获得Factor Class
    :return:
    """
    for f in FACTORS:
        _names = f(None, None).name
        if name in _names: return f
    raise ValueError(f"通过因子名称[{name}]无法找到因子类Class")
