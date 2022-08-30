# coding: utf-8

import logging
import os.path
import time

import pandas as pd
import statsmodels.formula.api as sm

from mlstock.factors.factor import ComplexMergeFactor
from mlstock.factors.fama import fama_model
from mlstock.utils import utils

logger = logging.getLogger(__name__)

"""

华泰金工：特质波动率——个股最近N个月内用日频收益率对 Fama-French 三因子回归的残差的标准差，N=1，3，6，12

这行注释删除：~~我没有用"日频"，我直接用的周频，日频的计算量太大了，周频也可以吧，反正就是一种特征工程。~~

我本来想用周频，后来发现算的是std标准差，用1个、3个、6个这样的数据数量根本不够算std啊，
所以，还是得切回到日频，当然，这样的话，计算量会大很多。

所谓"特质波动率"： 就是源于一个现象"低特质波动的股票，未来预期收益更高"。

参考：
- https://www.joinquant.com/view/community/detail/b27081ecc7bccfc7acc484f8a63e2459
- https://www.joinquant.com/view/community/detail/1813dae5165ee3c5c81e2408d7fe576f
- https://zhuanlan.zhihu.com/p/30158144
- https://zhuanlan.zhihu.com/p/379585598
- https://mp.weixin.qq.com/s/k_2ltrIQ7jkgAKhDc7Vo2A
- https://blog.csdn.net/FightingBob/article/details/106791144
- https://uqer.datayes.com/v3/community/share/58db552a6d08bb0051c52451

特质波动率(Idiosyncratic Volatility, IV)与预期收益率的负向关系既不符合经典资产定价理论，
也不符合基于不完全信息的定价理论，因此学术界称之为“特质波动率之谜”。

该因子虽在多头部分表现略逊于流通市值因子，但在多空方面表现明显强于流通市值因子，
说明特质波动率因子具有很好的选股区分能力，并且在空头部分有良好的风险警示作用。

基于CAPM的特质波动率 IVCAPM: 就是基于CAMP的残差的年化标准差来衡量。
基于Fama-French三因子模型的特质波动率 IVFF3： 就是在IVCAMP的基础上，再剔除市值因子和估值因子后的残差的年化标准差来衡量。

----
关于实现：

    特质波动率，可以有多种实现方法，可以是CAMP市场的残差，也可以是Fama-Frech的残差，这里，我用的是FF3的残差，
    啥叫FF3的残差，就是，用Fama定义的模型，先去算因子收益，[参考](../../fama/factor.py)中，
    使用股票池（比如中证500），去算出来的全市场的SMB，HML因子，
    然后，就可以对某一直股票，比如"招商银行"，对他进行回归：r_i = α_i + b1 * r_m_i + b2 * smb_i + b3 * hml_i + e_i
    我们要的就是那个e_i，也就是这期里，无法被模型解释的'**特质**'。上式计算，用的是每天的数据，为何强调这点呢，是为了说明e_i的i，指的是每天。
    那波动率呢？
    就是计算你回测周期的内的标准差 * sqrt(T)，比如你回测周期是20天，那就是把招商银行这20天的特异残差求一个标准差，然后再乘以根号下20。
    这个值，是这20天共同拥有的一个"特异波动率"，对，这20天的因子暴露值，都一样，都是这个数！
    我是这么理解的，也不知道对不对，这些文章云山雾罩地不说人话都。
"""

N = [1, 3, 6, 12]
WEEK_TRADE_DAYS = 5


class FF3ResidualStd(ComplexMergeFactor):

    @property
    def name(self):
        return [f"std_ff3factor_{i}w" for i in N]

    @property
    def cname(self):
        return [f'{i}周特异波动率' for i in N]

    def _calculate_one_stock_ff3_residual(self, df_one_stock_daily, df_fama, period):
        """
        计算特异性：讲人话，就是用fama计算后它（这只股票）的理论价格，和它的实际价格的偏差，
        周频的每周，都可以依据当周的df_fama因子（市场、smb、hml）算出每一个股票的理论价格，这3个因子是所有的股票共享的哈，牢记，
        所以，对一只股票而言，每周的df_fama因子（市场、smb、hml）不同，就导致用它算出的这只股票的特异性数值，
        这样下来，这只股票的每周的特异性数值（用fama因子计算出来的残差），就是一个时间序列。
        这个时间序列，就是我们所要的。

        "用横截面（每期）上的所有股票算出的smb/hml/rm，用其当做当期因子，然后对单只股票所有期，进行时间序列回归，可得系数权重标量，和每期残差"

        参考：https://zhuanlan.zhihu.com/p/131533515
        :param df_daily: 按照时间序列，所有股票共享的fama-french的三个因子，每周3个值，N期
        :param df_fama: 按照时间序列，所有股票共享的fama-french的三个因子，每周3个值，N期
        :return: 返回的一个每只股票的特异性收益率的时间序列
        """
        stock_code = df_one_stock_daily.name  # 保留一下股票名称，因为下面的merge后，就会消失
        # 细节：1个时间截面上，的所有股票，共享fama的3因子数值
        df_one_stock_daily = df_one_stock_daily.merge(df_fama, on=['trade_date'], how='left')

        def _calculate_residual_std(self, s):
            """
            :param df: 一只股票的当日之前的N天的数据
            :return:
            """
            df = df_one_stock_daily[s.index]

            # 细节：这个是这只股票的所有期（我们是N周）N*5天的数据进行回归
            ols_result = sm.ols(formula='pct_chg ~ R_M + SMB + HML', data=df).fit()

            # 获得残差，注意，这个是每个股票都每周，都计算出来一个残差来
            std = ols_result.resid.std()

            return std

        df_one_stock_daily.pct_chg.rolling(window=period).apply(lambda x: _calculate_residual_std(x), raw=False)

    def calculate(self, stock_data):
        """
        计算是以天为最小单位，1、3、6、12周的窗口长度的收益率波动标准差
        :param stock_data:
        :return:
        """

        df_weekly = stock_data.df_weekly
        df_weekly = df_weekly.sort_values(['ts_code', 'trade_date'])

        df_daily = stock_data.df_daily
        df_index_daily = stock_data.df_index_daily
        df_daily_basic = stock_data.df_daily_basic
        start_time = time.time()
        df_fama = fama_model.calculate_factors(df_stocks=df_daily, df_market=df_index_daily, df_basic=df_daily_basic)
        utils.time_elapse(start_time, "计算完市场的Fama-Frech三因子数据")

        # 2.按照要求计算以1、3、6、12周的滑动窗口，计算出每期的的特异性波动的方差
        start_time = time.time()
        df_residual_stds = []
        for i, n in enumerate(N):
            # 变成周频
            time_window = n * WEEK_TRADE_DAYS
            # 每只股票，针对"每周"数据，都要逐个计算其残差std
            df = df_weekly[['ts_code', 'trade_date', 'pct_chg']].groupby('ts_code') \
                .apply(self._calculate_one_stock_ff3_residual,
                       df_fama=df_fama,
                       period=time_window)
            df_residual_stds.append(df.reset_index(drop=True))
            start_time = utils.time_elapse(start_time,
                                           f"计算完时间窗口{i}周的所有股票Fama-French回归残差的标准差：{len(df)}行",
                                           "debug")

        df_residuals = pd.concat(df_residual_stds, axis=1)

        return df_residuals[['ts_code', 'trade_date'] + self.name]

    def _calculate_ff3_residual(self, stock_data):
        """
        获得各只股票的信息
        """
        df_daily = stock_data.df_daily
        df_index_daily = stock_data.df_index_daily
        df_daily_basic = stock_data.df_daily_basic

        # df_fama['trade_date', 'R_M', 'SMB', 'HML']
        # fama三因子，是 用每天的横截面'凑'出来的特征，用来后面回归单个股票的数据
        start_time = time.time()
        df_fama = fama_model.calculate_factors(df_stocks=df_daily, df_market=df_index_daily, df_basic=df_daily_basic)
        start_time = utils.time_elapse(start_time, f"计算完所有股票的日频Fama-French因子：{len(df_fama)}行")

        assert len(df_fama) > 0, "3因子数据行数应该大于0"

        """
        计算每只股票，和FF3的每天的残差
        
        先做Fmam-French的三因子回归：
            r_i = α_i + b1 * r_m_i + b2 * smb_i + b3 * hml_i + e_i 
            对每一只股票做回归，r_i,r_m_i，smb_i，hml_i 已知，这里的i表示的就是股票的序号，不是时间的序号哈，
            这里的r_i可不是一天，是所有人的日期，比如 回归的数据，是，招商银行，从2008年到2021年
            回归后，可以得到α_i、b1、b2、b3、e_i，我们这里只需要残差e_i，这里的残差也是多天的残差，这只股票的多天的残差。
        参考：
            - https://blog.csdn.net/CoderPai/article/details/82982146 
            - https://zhuanlan.zhihu.com/p/261031713
            使用statsmodels.formula中的ols，可以写一个表达式，来指定Y和X_i，即dataframe中的列名，很酷，喜欢        
        数据：
            合并后数据如下：
            trade_date  R_M     SMB         HML         ts_code     pct_chg
            2016-06-24	0.12321 0.165260	0.002198	0.085632    0.052
            2016-06-27	0.2331  0.165537	0.003583	0.063299    0.01
            2016-06-28	0.1234  0.135215	0.010403	0.059038    0.035
            ...
        做回归：
            r_i = α_i + b1 * r_m_i + b2 * smb_i + b3 * hml_i + e_i 
            某一只股票的所有的日子的数据做回归，r_i,r_m_i，smb_i，hml_i 已知，回归后，得到e_i(残差)
        """
        # as_index=True是为了保留ts_code，但是在ubuntu服务器上不行，排查后发现是pandas==1.4.3导致，应该是个bug
        # 解决办法是降级pandas，2022.8.11
        df_residuals = df_daily.groupby('ts_code', as_index=True).apply(self._calculate_one_stock_ff3_residual,
                                                                        df_fama).reset_index()
        utils.time_elapse(start_time, f"计算完所有股票的Fama-French回归残差：{len(df_residuals)}行")

        return df_residuals[['ts_code', 'trade_date', 'ff3_residual']]


# python -m mlstock.factors.ff3_residual_std
if __name__ == '__main__':
    from mlstock.data import data_filter
    from mlstock.data import data_loader, data_filter
    from mlstock.data.datasource import DataSource
    from mlstock.data.stock_info import StocksInfo

    utils.init_logger(file=False)

    start_time = time.time()

    start_date = "20180101"
    end_date = "20220801"
    df_stock_basic = data_filter.filter_stocks()
    df_stock_basic = df_stock_basic.iloc[:50]
    datasource = DataSource()
    stocks_info = StocksInfo(df_stock_basic.ts_code, start_date, end_date)
    df_stocks = data_loader.load(datasource, df_stock_basic.ts_code, start_date, end_date)

    factor_alpha_beta = FF3ResidualStd(datasource, stocks_info)
    df = factor_alpha_beta.calculate(df_stocks)

    print("因子结果\n", df)
    utils.time_elapse(start_time, "全部处理")
