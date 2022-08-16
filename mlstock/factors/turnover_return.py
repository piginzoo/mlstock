import logging

from mlstock.factors.factor import ComplexMergeFactor


logger = logging.getLogger(__name__)

N = [1, 3, 6, 12]


class TurnoverReturn(ComplexMergeFactor):
    """

    华泰金工原始：`动量反转 wgt_return_Nm:个股最近N个月内用每日换手率乘以每日收益率求算术平均值，N=1，3，6，12`
    我们这里就是N周，
    注意，这里是"每日"，所以我们需要加载每日，<--------- **每日**

    daily_basic是每日的数据，我们目前是每周的数据，
    神仔的做法是：
        df['wgt_return_1m'] = df['close'] * df['pct_chg']
        df = df.groupby(idx_f).wgt_return_1m.agg('mean')

    我觉得没必要乘以close收盘价啊。
    """

    @property
    def name(self):
        return [f'turnover_return_{i}w' for i in N]

    @property
    def cname(self):
        return [f'{i}周内每日换手率乘以每日收益率的算术平均值' for i in N]

    def calculate(self, stock_data):
        df_daily = stock_data.df_daily
        df_daily_basic = stock_data.df_daily_basic

        """ https://tushare.pro/document/2?doc_id=32
        code        datetime  turnover_rate_f
        600230.SH   20180726  4.5734         
        """
        df_daily_basic = df_daily_basic[['trade_date', 'ts_code', 'turnover_rate_f']]
        df_daily = df_daily[['trade_date', 'ts_code', 'pct_chg']]

        """
        2022.8.16 一个低级但是查了半天才找到的bug，
        数据不做下面的排序，默认是按照trade_date+ts_code的顺序排序的，
        会导致下面的赋值出现问题：
            df[f'turnover_return_{i}w'] = df.groupby('ts_code').turnover_return.rolling(i * 5).mean().values
        改为
            df[f'turnover_return_{i}w'] = df.groupby('ts_code').turnover_return.rolling(i * 5).mean().reset_index(level=0, drop=True)
        就没有问题，原因是后者是一个Series，有索引，可以和原有的序列对上，
        而前者只是一个numpy array，与之前的  trade_date+ts_code 的顺序对齐的话，当然乱掉了
        所以，只要这里做了排序，用numpy array还是Series，都无所谓嘞
        这个bug查了我1天，唉，低级错误害死人 
        """
        df = df_daily.merge(df_daily_basic, on=['ts_code', 'trade_date'])

        # `动量反转 wgt_return_Nm:个股最近N个月内用每日换手率乘以每日收益率求算术平均值，N=1，3，6，12`
        df['turnover_return'] = df['turnover_rate_f'] * df['pct_chg']

        df = df.sort_values(['ts_code', 'trade_date'])

        for i in N:
            # x5，按照每周交易日5天计算的
            # 我靠！隐藏的很深的一个bug，找各种写法会导致中间莫名其妙的出现nan，而且计算的也不对，改为后者就ok了
            # df[f'turnover_return_{i}w'] = df.groupby('ts_code').turnover_return.rolling(i * 5).mean().values
            df[f'turnover_return_{i}w'] = df.groupby('ts_code').turnover_return.rolling(i * 5).mean().reset_index(level=0, drop=True)

        # 返回ts_code和trade_date是为了和周频数据做join
        return df[['trade_date', 'ts_code'] + self.name]

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
    - turnover_Nm：个股最近N个月的日均换手率，表现了个股N个月内的流动性水平。N=1,3,6
    - turnover_bias_Nm：个股最近N个月的日均换手率除以个股两年内日均换手率再减去1，代表了个股N个月内流动性的乖离率。N=1,3,6
    - turnover_std_Nm：个股最近N个月的日换手率的标准差，表现了个股N个月内流动性水平的波动幅度。N=1,3,6
    - turnover_bias_std_Nm：个股最近N个月的日换手率的标准差除以个股两年内日换手率的标准差再减去1，代表了个股N个月内流动性的波动幅度的乖离率。N=1,3,6
    """
# python -m mlstock.factors.turnover_return
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
    stocks = ['000401.SZ']
    datasource = DataSource()
    stocks_info = StocksInfo(stocks, start_date, end_date)
    stock_data = data_loader.load(datasource, stocks, start_date, end_date)
    # 把基础信息merge到周频数据中
    df_stock_basic = datasource.stock_basic(stocks)
    df_weekly = stock_data.df_weekly.merge(df_stock_basic, on='ts_code', how='left')

    tr = TurnoverReturn(datasource, stocks_info)
    df_factors = tr.calculate(stock_data)
    df = tr.merge(df_weekly,df_factors)
    df = df[df.trade_date>start_date]
    print(df)
    print("NA缺失比例", df.isna().sum() / len(df))
    print(df[(df.ts_code=='000401.SZ')&(df.trade_date=='20180629')])
