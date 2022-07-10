import backtrader as bt
from backtrader import Indicator


class KDJ(Indicator):
    """
    自定义KDJ指标，指标计算方法如下：
    RSV = （收盘价-N周期最低价）/（N周期最高价-N周期最低价）*100
        （注意，区分RSI = 上升的平均价格/(上升平均+下降平均）*100 )
    K值 = RSV的N周期加权移动平均值(EMA)
    D值 = K值的N周期加权移动平均值(EMA)
    J值 = 3_K-2_D
    """

    lines = ('K', 'D', 'J') #
    params = (('period_signal', 9),)

    def __init__(self):
        super(KDJ, self).__init__()

        high = bt.indicators.Highest(self.data.high, period=self.p.period_signal) # 9天的最高价
        low = bt.indicators.Lowest(self.data.low, period=self.p.period_signal) # 9天的最低价
        RSV = 100 * bt.DivByZero(self.data_close - low, high - low, zero=None)
        self.lines.K = bt.indicators.EMA(RSV, period=5)  # 5 日平滑移动平均线，感觉才和同花顺的吻合上
        self.lines.D = bt.indicators.EMA(self.lines.K, period=5)
        self.lines.J = 3 * self.lines.K - 2 * self.lines.D