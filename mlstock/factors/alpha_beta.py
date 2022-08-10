"""
HAlpha 个股 60 个月收益与上证综指回归的截距项,
对我周频来说，就用60个周的上证综指回归的截距项,

beta : 个股 60 个月收益与上证综指回归的beta

回归使用的是CAPM公式：
R_it - r_f = alpha_it + beta_it * (R_mt - r_f) + e_it
个股收益 - 无风险收益 = 个股alpha + 个股beta * (市场/上证综指收益 - 无风险收益) + 个股扰动项
- i：是股票序号；
- t：是一个周期，我们这里是周；
- e：epsilon，扰动项

一般情况，是用N个周期的市场（上证综指）收益，和N个个股收益，一共是N对数据（R_it,R_mt)，来回归alpha_it和beta_it的，
现在呢，华泰金工要求我们用60周的，实际上你可以想象是一个滑动时间窗口，60周，就是N=60，
用这60个周的这只股票的收益率，和60个周的市场（上证综指）收益率，回归出2个数：alpha和beta这两个数。
也就是你站在当期，向前倒腾60周，回归出来的。

然后，然后，你往下周移动一下，相当于时间窗口移动了一下，那么就又有新的60对数据（R_it,R_mt)出来了，
当然，其中59个和上一样的，但是，这个时候，你再回归，就又会得到一个alpha和beta，

这样，每一期，都要做这么一个回归，都得到一个beta，一个aphla，这个就是当期的华泰说的HAlpha和beta。

实现的时候，用了2个apply，每周五，都向前回溯60周，然后用这60周的数据回归alpha和beta

"""
from mlstock.factors.factor import SimpleFactor, ComplexMergeFactor
from mlstock.utils import utils
import numpy as np


class AlphaBeta(ComplexMergeFactor):
    # 英文名
    @property
    def name(self):
        return ["alpha", "beta"]

    # 中文名
    @property
    def cname(self):
        return ["alpha", "beta"]

    def _handle_stock(self, df_stock_weekly):
        """
        处理一只股票，用60周的滑动窗口的概念，来不断地算每天，向前60天的数据回归出来的α和β（CAPM的概念）
        参考： https://www.jianshu.com/p/1eaf89990ce7
        apply返回多列的时候，只能通过axis=1 + result_type="expand"，来处理，
        且，必须是整行apply，不能单列apply（df_stock_weekly.trade_date.apply(...)，这种axis=1就报错！）
        这块搞了半天，靠！
        """
        return df_stock_weekly.apply(self._handle_60_weeks_OLS,
                                     df_stock_weekly=df_stock_weekly,
                                     axis=1,
                                     result_type="expand")

    def _handle_60_weeks_OLS(self, date, df_stock_weekly):
        # 取得当周的日期（周最后一天）
        date = date['trade_date']
        df_recent_60 = df_stock_weekly[df_stock_weekly['trade_date'] <= date][-60:]
        # 太少的回归不出来
        if len(df_recent_60) < 2: return np.nan, np.nan

        X = df_recent_60['pct_chg'].values
        y = df_recent_60['pct_chg_index'].values

        # 用这60周的60个数据，做线性回归，X是个股的收益，y是指数收益，求出截距和系数，即alpha和beta
        params, _ = utils.OLS(X, y)

        alpha, beta = params[0], params[1]

        # if np.isnan(alpha):
        #     import pdb;pdb.set_trace()
        #     print(date,df_stock_weekly[df_stock_weekly['trade_date'] == date],y)
        #     print("-"*40)
        return alpha, beta

    def calculate(self, stock_data):
        df_weekly = stock_data.df_weekly
        df_index_weekly = stock_data.df_index_weekly
        df_index_weekly = df_index_weekly.rename(columns={'pct_chg': 'pct_chg_index'})
        df_index_weekly = df_index_weekly[['trade_date', 'pct_chg_index']]

        df_weekly = df_weekly.merge(df_index_weekly, on=['trade_date'], how='left')
        # 2022.8.10，bugfix，股票非周五导致weekly指数收益为NAN，导致其移动平均为NAN，导致大量数据缺失，因此需要drop掉这些异常数据
        df_weekly.dropna(subset=['pct_chg_index'], inplace=True)

        # 先统一排一下序
        df_weekly = df_weekly.sort_values(['ts_code', 'trade_date'])
        df_weekly[['alpha', 'beta']] = df_weekly.groupby(['ts_code']).apply(self._handle_stock)
        return df_weekly[['ts_code', 'trade_date', 'alpha', 'beta']]


# python -m mlstock.factors.alpha_beta
if __name__ == '__main__':
    utils.init_logger(file=False)

    start_date = "20180101"
    end_date = "20220101"
    stocks = ['000017.SZ']  # '600000.SH', '002357.SZ', '000404.SZ', '600230.SH']

    from mlstock.data import data_loader
    from mlstock.data.datasource import DataSource
    from mlstock.data.stock_info import StocksInfo

    datasource = DataSource()
    stocks_info = StocksInfo(stocks, start_date, end_date)
    df_stocks = data_loader.load(datasource, stocks, start_date, end_date)

    print(df_stocks.df_weekly.count())
    factor_alpha_beta = AlphaBeta(datasource, stocks_info)
    df = factor_alpha_beta.calculate(df_stocks)
    print(df[df['beta'].isna()])
    print(df.count())
