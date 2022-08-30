"""
换手率因子：

- https://zhuanlan.zhihu.com/p/37232850
- https://crm.htsc.com.cn/doc/2017/10750101/6678c51c-a298-41ba-beb9-610ab793cf05.pdf  华泰~换手率类因子
- https://uqer.datayes.com/v3/community/share/5afd527db3a1a1012acad84c

换手率因子是一类很重要的情绪类因子，反映了一支股票在一段时间内的流动性强弱，和持有者平均持有时间的长短。
一般来说换手率因子的大小和股票的收益为负向关系，即换手率越高的股票预期收益越低，换手率越低的股票预期收益越高。

四个构造出来的换手率类的因子（都是与股票的日均换手率相关）：
- turnover_Nm：个股最近N周的日均换手率，表现了个股N周内的流动性水平。N=1,3,6
- turnover_bias_Nm：个股最近N周的日均换手率除以个股两年内日均换手率再减去1，代表了个股N周内流动性的乖离率。N=1,3,6
- turnover_std_Nm：个股最近N周的日换手率的标准差，表现了个股N周内流动性水平的波动幅度。N=1,3,6
- turnover_bias_std_Nm：个股最近N周的日换手率的标准差除以个股两年内日换手率的标准差再减去1，代表了个股N周内流动性的波动幅度的乖离率。N=1,3,6

这是4个因子哈，都是跟换手率相关的，他们之间具备共线性，是相关的，要用的时候，挑一个好的，或者，做因子正交化后再用。

市值中性化：换手率类因子与市值类因子存在一定程度的负相关性，我们对换手率因子首先进行市值中性化处理，从而消除了大市值对于换手率因子表现的影响。

知乎文章的结论：进行市值中性化处理之后，因子表现有明显提高。在本文的回测方法下，turnover_1m和turnover_std_1m因子表现较好。
"""

import logging

from mlstock.factors.factor import SimpleFactor

logger = logging.getLogger(__name__)

N = [1, 3, 6, 12]


class Turnover(SimpleFactor):
    """
    个股最近N个月内 日均换手率 剔除停牌 涨跌停的交易
    """

    @property
    def name(self):
        return [f'turnover_{i}w' for i in N] + \
               [f'turnover_bias_{i}w' for i in N] + \
               [f'turnover_std_{i}w' for i in N] + \
               [f'turnover_bias_std_{i}w' for i in N]

    @property
    def cname(self):
        return [f'{i}周换手率' for i in N] + \
               [f'{i}周换手率偏差' for i in N] + \
               [f'{i}周换手率标准差' for i in N] + \
               [f'{i}周换手率标准差偏差' for i in N]

    def calculate(self, stock_data):
        df_daily_basic = stock_data.df_daily_basic

        """
        # https://tushare.pro/document/2?doc_id=32
                    code  datetime  turnover_rate_f       circ_mv
        0     600230.SH   20180726           4.5734  1.115326e+06
        1     600237.SH   20180726           1.7703  2.336490e+05
        """
        df_daily_basic = df_daily_basic[['ts_code', 'trade_date', 'turnover_rate_f', 'circ_mv']]
        df_daily_basic.columns = ['ts_code', 'trade_date', 'turnover_rate', 'circ_mv']
        df_daily_basic = df_daily_basic.sort_values(['ts_code', 'trade_date'])

        datas = self.calculate_turnover_rate(df_daily_basic)

        return datas

    """
    nanmean:忽略nan，不参与mean，例：
        >>> a = np.array([[1, np.nan], [3, 4]])
        >>> np.nanmean(a)
        2.6666666666666665
        >>> np.nanmean(a, axis=0)
        array([2.,  4.])
        >>> np.nanmean(a, axis=1)
        array([1.,  3.5]) # may vary
    rolling:
        https://blog.csdn.net/maymay_/article/details/80241627
    # 定义因子计算逻辑
    - turnover_Nm：个股最近N周的日均换手率，表现了个股N周内的流动性水平。N=1,3,6
    - turnover_bias_Nm：个股最近N周的日均换手率除以个股两年内日均换手率再减去1，代表了个股N周内流动性的乖离率。N=1,3,6
    - turnover_std_Nm：个股最近N周的日换手率的标准差，表现了个股N周内流动性水平的波动幅度。N=1,3,6
    - turnover_bias_std_Nm：个股最近N周的日换手率的标准差除以个股两年内日换手率的标准差再减去1，代表了个股N周内流动性的波动幅度的乖离率。N=1,3,6
    """

    def calculate_turnover_rate(self, df):
        """
        注意，df是日频数据，不是周频的，特别指明一下，所以要windows=5*i（N周）
        英文命名上有歧义，还是按照中文解释为主
        :param df:
        :return:
        """

        # 24周内的均值和标准差值，这个不是指标，是用于计算指标用的中间值
        df[f'turnover_24w'] = df.groupby('ts_code')['turnover_rate'].rolling(window=5 * 24,
                                                                             min_periods=1).mean().values
        df[f'turnover_std_24w'] = df.groupby('ts_code')['turnover_rate'].rolling(window=5 * 24,
                                                                                 min_periods=1).std().values

        # 1.N周的日换手率均值
        for i in N:
            df[f'turnover_{i}w'] = df.groupby('ts_code')['turnover_rate'].rolling(window=5 * i,
                                                                                  min_periods=1).mean().values
        # 2.N周的日换手率 / 两年内日换手率 - 1，表示N周流动性的乖离率
        # import pdb; pdb.set_trace()
        for i in N:
            df[f'turnover_bias_{i}w'] = df.groupby('ts_code').apply(
                lambda df_stock: df_stock[f'turnover_{i}w'] / df_stock.turnover_24w - 1).values

        # 3.N周的日均换手率的标准差
        for i in N:
            df[f'turnover_std_{i}w'] = df.groupby('ts_code')['turnover_rate'].rolling(window=5,
                                                                                      min_periods=2).std().values

        # N周的日换手率的标准差 / 两年内日换手率的标准差 - 1，表示N周波动幅度的乖离率
        for i in N:
            df[f'turnover_bias_std_{i}w'] = df.groupby('ts_code').apply(
                lambda df_stock: df_stock[f'turnover_std_{i}w'] / df_stock.turnover_std_24w - 1).values

        return df[self.name]


# python -m mlstock.factors.turnover
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
    stocks = ['000401.SZ','600000.SH', '002357.SZ', '000404.SZ', '600230.SH']
    datasource = DataSource()
    stocks_info = StocksInfo(stocks, start_date, end_date)
    stock_data = data_loader.load(datasource, stocks, start_date, end_date)
    # 把基础信息merge到周频数据中
    df_stock_basic = datasource.stock_basic(stocks)
    df_weekly = stock_data.df_weekly.merge(df_stock_basic, on='ts_code', how='left')

    tr = Turnover(datasource, stocks_info)
    df_factors = tr.calculate(stock_data)
    df = tr.merge(df_weekly, df_factors)
    df = df[df.trade_date > start_date]
    print(df)
    print("NA缺失比例", df.isna().sum() / len(df))