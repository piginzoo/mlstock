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
           FF3ResidualStd,
           AlphaBeta,
           StakeHolder]

