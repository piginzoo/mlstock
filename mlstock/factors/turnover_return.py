import logging

from mlstock.factors.factor import ComplexMergeFactor

logger = logging.getLogger(__name__)

N = [1, 3, 6, 12]


class TurnoverReturn(ComplexMergeFactor):
    """

    华泰金工原始：`动量反转 wgt_return_Nm:个股最近N个月内用每日换手率乘以每日收益率求算术平均值，N=1，3，6，12`
    我们这里就是N周，
    注意，这里是"每日"，所以我们需要加载每日，

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
        code        datetime  turnover_rate_f  circ_mv
        600230.SH   20180726  4.5734           1.115326e+06
        """
        df_daily_basic = df_daily_basic[['trade_date', 'ts_code', 'turnover_rate_f']]
        df_daily = df_daily[['trade_date', 'ts_code', 'pct_chg']]

        df = df_daily.merge(df_daily_basic, on=['ts_code', 'trade_date'])

        # `动量反转 wgt_return_Nm:个股最近N个月内用每日换手率乘以每日收益率求算术平均值，N=1，3，6，12`
        df['turnover_return'] = df['turnover_rate_f'] * df['pct_chg']

        for i in N:
            # x5，按照每周交易日5天计算的
            df[f'turnover_return_{i}w'] = df.groupby('ts_code').turnover_return.rolling(i * 5).mean().values

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
