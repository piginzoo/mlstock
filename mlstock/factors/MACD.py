import talib as ta

from .factor import Factor

# macd 30日
# dea 10日
# dif
fastperiod = 10
slowperiod = 30
signalperiod = 9


class MACD(Factor):

    # 英文名
    @property
    def name(self):
        return "MACD"

    # 中文名
    @property
    def cname(self):
        return "MACD"

    def calculate(self, df):
        return self.__macd(df.close)

    def __macd(self, x):
        macd, dea, dif = ta.MACD(x,
                                 fastperiod=fastperiod,
                                 slowperiod=slowperiod,
                                 signalperiod=signalperiod)
        return macd
