import talib

from mlstock.factors.factor import SimpleFactor

PERIOD = 20


class RSI(SimpleFactor):
    """
    相对强弱指标RSI是用以计测市场供需关系和买卖力道的方法及指标。
    计算公式：
    N日RSI =A/（A+B）×100
    A=N日内收盘涨幅之和
    B=N日内收盘跌幅之和（取正值）
    由上面算式可知RSI指标的技术含义，即以向上的力量与向下的力量进行比较，
    若向上的力量较大，则计算出来的指标上升；若向下的力量较大，则指标下降，由此测算出市场走势的强弱。
    """

    # 英文名
    @property
    def name(self):
        return "RSI"

    # 中文名
    @property
    def cname(self):
        return "RSI"

    def calculate(self, stock_data):
        df_weekly = stock_data.df_weekly
        return self.rsi(df_weekly.close, period=PERIOD)

    # psy 20日
    def rsi(self, x, period=PERIOD):
        return talib.RSI(x, timeperiod=period)
