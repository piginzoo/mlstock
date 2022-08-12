import talib as ta
from .factor import SimpleFactor

fastperiod = 12
slowperiod = 26
signalperiod = 9

class MACD(SimpleFactor):
    """
    1、线1：先得到一个DIF ： EMA12 - EMA26
    2、线2：在得到一个DEA：DIF的9日加权移动平均
    所以，26周 + 9周 = 35周，才可能得到一个有效的dea值，所以要预加载35周，大约9个月的数据
    """

    # 英文名
    @property
    def name(self):
        return "MACD"

    # 中文名
    @property
    def cname(self):
        return "MACD"

    def calculate(self, stock_data):
        df_weekly = stock_data.df_weekly
        return df_weekly.groupby('ts_code').close.apply(self.__macd)

    def __macd(self, x):
        macd, dea, dif = ta.MACD(x,
                                 fastperiod=fastperiod,
                                 slowperiod=slowperiod,
                                 signalperiod=signalperiod)
        return macd


# python -m mlstock.factors.macd
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
    stocks = ['000007.SZ ']
    datasource = DataSource()
    stocks_info = StocksInfo(stocks, start_date, end_date)
    stock_data = data_loader.load(datasource, stocks, start_date, end_date)
    df = MACD(datasource, stocks_info).calculate(stock_data)
    df_weekly = stock_data.df_weekly
    valid_len = len(df_weekly[df_weekly['trade_date']>start_date])
    df = df[-valid_len:]
    print(df)
    print("NA占比: %.2f%%" % (df.isna().sum()*100/len(df)))
    import pdb;pdb.set_trace()