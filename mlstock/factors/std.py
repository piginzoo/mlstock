import logging

from mlstock.factors.factor import SimpleFactor
import pandas as pd

logger = logging.getLogger(__name__)

"""
波动率因子：
https://zhuanlan.zhihu.com/p/30158144
波动率因子有很多，我这里的是std，标准差，
而算标准差，又要设置时间窗口
"""

mapping = [
    {'name': 'std_1w', 'cname': '1周波动率', 'period': 1},
    {'name': 'std_3w', 'cname': '3周波动率', 'period': 3},
    {'name': 'std_6w', 'cname': '6周波动率', 'period': 6},
    {'name': 'std_12w', 'cname': '12周波动率', 'period': 12}
]


class Std(SimpleFactor):

    @property
    def name(self):
        return [m['name'] for m in mapping]

    @property
    def cname(self):
        return [m['cname'] for m in mapping]

    def calculate(self, stock_data):
        df_weekly = stock_data.df_weekly
        """
        计算波动率，波动率，就是往前回溯period个周期
        """
        results = []
        for m in mapping:
            # x 5，是近似于周的频度（5个交易日是一个周）
            df_std = df_weekly.pct_chg.rolling(window=m['period']*5).std()
            results.append(df_std)
        df = pd.concat(results, axis=1)  # 按照列拼接（axis=1）
        return df
