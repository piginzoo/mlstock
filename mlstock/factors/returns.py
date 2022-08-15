import logging

import numpy as np
import pandas as pd

from mlstock.factors.factor import SimpleFactor

logger = logging.getLogger(__name__)

"""
个股最近N个周收益率 N=1, 3, 6, 12，
注意，是累计收益
"""

N = [1, 3, 6, 12]


class Return(SimpleFactor):

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
            df_return = df_weekly.groupby('ts_code').pct_chg.apply(lambda x: self._calculte_return_N(x, period))
            results.append(df_return)
        df = pd.concat(results, axis=1)  # 按照列拼接（axis=1）
        return df

# python -m mlstock.factors.returns
if __name__ == '__main__':
    from mlstock.data import data_loader
    from mlstock.data.datasource import DataSource
    from mlstock.data.stock_info import StocksInfo
    from mlstock.utils import utils
    import pandas
    pandas.set_option('display.max_rows', 1000000)
    utils.init_logger(file=False)

    start_date = "20180101"
    end_date = "20200101"
    stocks = ['000401.SZ']
    datasource = DataSource()
    stocks_info = StocksInfo(stocks, start_date, end_date)
    stock_data = data_loader.load(datasource, stocks, start_date, end_date)
    df = Return(datasource, stocks_info).calculate(stock_data)
    # df = df[df.trade_date>start_date]
    print(df)
    print("NA缺失比例", df.isna().sum() / len(df))
