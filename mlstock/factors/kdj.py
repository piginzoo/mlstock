import talib
import pandas as pd
from mlstock.factors.factor import SimpleFactor

fastk_period = 9
slowk_period = 3
slowk_matype = 0
slowd_period = 3
slowd_matype = 0


class KDJ(SimpleFactor):
    # 英文名
    @property
    def name(self):
        return "KDJ"

    # 中文名
    @property
    def cname(self):
        return "KDJ"

    def calculate(self, stock_data):
        df_weekly = stock_data.df_weekly
        K, D = talib.STOCH(
            df_weekly.high,
            df_weekly.low,
            df_weekly.close,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=slowk_matype,
            slowd_period=slowd_period,
            slowd_matype=slowd_matype)

        # 求出J值，J = (3*K)-(2*D)
        J = pd.Series(list(map(lambda x, y: 3 * x - 2 * y, K, D)))
        return J
