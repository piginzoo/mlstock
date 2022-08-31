import numpy as np
import pandas as pd

from mlstock.factors.factor import SimpleFactor

N = [1, 3, 6, 12]


class PSY(SimpleFactor):
    """
    PSY: 心理线指标、大众指标，研究投资者心理波动的情绪指标。
    PSY = N天内上涨天数 / N * 100，N一般取12，最大不超高24，周线最长不超过26
    PSY大小反映市场是倾向于买方、还是卖方。
    """

    # 英文名
    @property
    def name(self):
        return ['PSY_{}w'.format(i) for i in N]

    # 中文名
    @property
    def cname(self):
        return ['PSY_{}w'.format(i) for i in N]

    def calculate(self, stock_data):
        df_daily = stock_data.df_daily

        df_daily = df_daily.sort_values(['ts_code', 'trade_date'])  # 默认是升序排列

        for i, n in enumerate(N):
            # 先根据收益率，转变成涨1，跌0
            df_daily['winloss'] = df_daily.pct_chg.apply(lambda x: x > 0)

            # 按股票来分组，然后每只股票做按照N周滚动计算PSY
            # 完事后要做reset_index(level=0,drop=True),原因是groupby+rolling后得到的df的index是（ts_code,原id）
            # 所以要drop掉ts_code，即level=0，这样才可以直接赋值回去（按照id直接对回原dataframe）
            # 全TMD是细节啊，我真的是服了pandas库了
            df = df_daily.groupby('ts_code')['winloss'].rolling(window=5 * n). \
                apply(lambda x: x.sum() / (5 * n)).reset_index(level=0, drop=True)
            df_daily[self.name[i]] = df
        return df_daily[self.name]


# python -m mlstock.factors.psy
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
    stocks = ['000401.SZ', '600000.SH', '002357.SZ']
    datasource = DataSource()
    stocks_info = StocksInfo(stocks, start_date, end_date)
    stock_data = data_loader.load(datasource, stocks, start_date, end_date)

    df = PSY(datasource, stocks_info).calculate(stock_data)
    print(df)
    print("NA缺失比例", df.isna().sum() / len(df))
