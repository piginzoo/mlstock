from mlstock.factors.factor import Factor
import numpy as np
import pandas as pd
PERIOD = 20


class PSY(Factor):
    """
    PSY: 心理线指标、大众指标，研究投资者心理波动的情绪指标。
    PSY = N天内上涨天数 / N * 100，N一般取12，最大不超高24，周线最长不超过26
    PSY大小反映市场是倾向于买方、还是卖方。
    """

    def __init__(self):
        self.datasource = datasource_factory.create(CONF['datasource'])

    # 英文名
    def name(self):
        return "PSY"

    # 中文名
    def cname(self):
        return "PSY"

    def calculate(self, stock_codes, start_date, end_date):
        pass

    # psy 20日
    def psy(x, period=20):
        difference = x[1:].values - x[:-1].values
        difference_dir = np.where(difference > 0, 1, 0)
        p = np.zeros((len(x),))
        p[:period] *= np.nan
        for i in range(period, len(x)):
            p[i] = (difference_dir[i - period + 1:i + 1].sum()) / period
        return pd.Series(p * 100, index=x.index)
