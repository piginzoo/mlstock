import talib as ta

from .factor import SimpleFactor

fastperiod = 12
slowperiod = 26
signalperiod = 9

class MACD(SimpleFactor):
    """
    1、线1：先得到一个DIF ： EMA12 - EMA26
    2、线2：在得到一个DEA：DIF的9日加权移动平均
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

