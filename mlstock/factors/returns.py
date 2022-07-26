import logging

import numpy as np
import pandas as pd

from mlstock.factors.factor import CommonFactor

logger = logging.getLogger(__name__)

"""
个股最近N个周收益率 N=1, 3, 6, 12，
注意，是累计收益
"""

N = [1, 3, 6, 12]


class Return(CommonFactor):

    @property
    def name(self):
        return ['return_{}w'.format(i) for i in N]

    @property
    def cname(self):
        return ['{}周累计收益'.format(i) for i in N]

    def _calculte_return_N(self, x, period):
        """计算累计收益，所以用 **prod** 乘法"""
        return (1 + x).rolling(window=period).apply(np.prod, raw=True) - 1

    def calculate(self, stock_data):
        df_weekly = stock_data.df_weekly
        """
        计算N周累计收益，就是往前回溯period个周期
        """
        results = []
        for period in N:
            df_return = df_weekly.pct_chg.apply(lambda x: self._calculte_return_N(x, period))
            results.append(df_return)
        return pd.concat(results, axis=1)  # 按照列拼接（axis=1）
