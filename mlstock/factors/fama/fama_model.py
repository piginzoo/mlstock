import logging

import pandas as pd

logger = logging.getLogger(__name__)

"""
Fama-French 3 因子，是用3个因子（市值、SMB、HML）来解释收益率，
因此，这个处理类的：
    输入是：2000只股票的各自收益率，市场收益率（用上证指数），还有市值和账面市值比的信息
    输出是：3个因子（市值、SMB、HML），以及3因子拟合股票收益率后的残差
再大致说一下FF3的思路：
    1、对于每一期，2000只股票（假设的），用他们构建SMB和HML，就是2数，市值因子就用上证收益率
    2、然后有N期，就有(N,3)个数，这就是X，而y是每只股票的N期收益率（N,1)
    3、对每只股票，做N期收益率 和 FF3（N，3）的回归
    4、对每只股票，回归出截距项、3个因子的系数，这是4个系数，是常数，
       还有一个长度为N的时间序列，即预测值(FF3拟合出来的)和真实值(股票N期每期的真实收益)的N期残差（N,1)
"""


# TODO: 我没有减去无风险收益，要么去读一下国债1年期收益周化，要么给一个固定的周化收益率，都可以，回头在改进


# %%定义计算函数
def calculate_smb_hml(df):
    """"
    参考：
    - https://zhuanlan.zhihu.com/p/55071842
    - https://zhuanlan.zhihu.com/p/341902943
    - https://zhuanlan.zhihu.com/p/21449852
    R_i = a_i + b_i * R_M + s_i * E(SMB) + h_i E(HML) + e_i
    - R_i：是股票收益率
    - SMB：市值因子，用的就是市值信息, circ_mv
        SMB = (SL+SM+SH)/3 - (BL+BM+BH)/3
    - HML：账面市值比，B/M，1/pb (PB是市净率=总市值/净资产)
        HML = (BH+SH)/2 - (BL+SL)/2
    """

    # 划分大小市值公司
    median = df['circ_mv'].median()
    df['SB'] = df['circ_mv'].map(lambda x: 'B' if x >= median else 'S')

    # 求账面市值比：PB的倒数
    df['BM'] = 1 / df['pb']
    # 划分高、中、低账面市值比公司
    border_down, border_up = df['BM'].quantile([0.3, 0.7])
    df['HML'] = df['BM'].map(lambda x: 'H' if x >= border_up else 'M')
    df['HML'] = df.apply(lambda row: 'L' if row['BM'] <= border_down else row['HML'], axis=1)

    # 组合划分为6组
    df_SL = df.query('(SB=="S") & (HML=="L")')
    df_SM = df.query('(SB=="S") & (HML=="M")')
    df_SH = df.query('(SB=="S") & (HML=="H")')
    df_BL = df.query('(SB=="B") & (HML=="L")')
    df_BM = df.query('(SB=="B") & (HML=="M")')
    df_BH = df.query('(SB=="B") & (HML=="H")')

    """
    # 计算各组收益率, pct_chg:涨跌幅 , circ_mv:流通市值（万元）
    # 以SL为例子：Small+Low
    #    小市值+低账面市值比，的一组，比如100只股票，把他们的当期收益"**按照市值加权**"后，汇总到一起
    #    每期，得到的SL是一个数，
    # 除100是因为pct_chg是按照百分比计算的，比如pct_chg=3.5，即3.5%，即0.035
    # 组内按市值赋权平均收益率 = sum(个股收益率 * 个股市值/组内总市值)
    """
    R_SL = ((df_SL['pct_chg'] / 100) * (df_SL['circ_mv'] / df_SL['circ_mv'].sum())).sum()  # 这种写法和下面的5种结果一样
    R_SM = (df_SM['pct_chg'] * df_SM['circ_mv'] / 100).sum() / df_SM['circ_mv'].sum()  # 我只是测试一下是否一致，
    R_SH = (df_SH['pct_chg'] * df_SH['circ_mv'] / 100).sum() / df_SH['circ_mv'].sum()  # 大约在千分之几，也对，我做的是每日的收益率
    R_BL = (df_BL['pct_chg'] * df_BL['circ_mv'] / 100).sum() / df_BL['circ_mv'].sum()
    R_BM = (df_BM['pct_chg'] * df_BM['circ_mv'] / 100).sum() / df_BM['circ_mv'].sum()
    R_BH = (df_BH['pct_chg'] * df_BH['circ_mv'] / 100).sum() / df_BH['circ_mv'].sum()

    # 计算SMB, HML并返回
    # 这个没啥好说的，即使按照Fama造的公式，得到了smb，smb是啥？是当期的一个数
    smb = (R_SL + R_SM + R_SH - R_BL - R_BM - R_BH) / 3
    hml = (R_SH + R_BH - R_SL - R_BL) / 2
    return smb, hml  # R_SL, R_SM, R_SH, R_BL, R_BM, R_BH


def calculate_factors(df_stocks, df_market, df_basic):
    """
    计算因子系数（不是因子），和，残差
    :param df_stocks: 股票的每期收益率
    :param df_market: 市场（上证指数）的每期收益率
    :param df_daily_basic: 每只股票的每期的基本信息
    :return: 每只股票的4个系数(N,4)，和拟合残差(N,M) N为股票数，M为周期数
    """

    # 获得股票池
    logger.debug("FF3计算：%d个股票", len(df_stocks))

    # 获取该日期所有股票的基本面指标，里面有市值信息
    df_stocks = df_stocks.merge(df_basic['ts_code', 'trade_date', 'circ_mv', 'pb'],  # circ_mv:市值，pb：账面市值比
                                on=['ts_code', 'trade_date'], how='left')

    df_market = df_market[['ts_code', 'trade_date', 'pct_chg']]
    df_market = df_market.rename(columns={'pct_chg', 'R_M'})
    df_stocks = df_stocks.merge(df_market['ts_code', 'trade_date', 'pct_chg'],
                                on=['ts_code', 'trade_date'], how='left')

    df_stocks[['SMB', 'HML']] = df_stocks.groupby(['ts_code', 'trade_date']).apply(calculate_smb_hml).reset_index()

    return df_stocks[['ts_code', 'trade_date', 'R_M', 'SMB', 'HML','pct_chg']]


# python -m fama.factor
if __name__ == '__main__':
    # index_code="000905.SH"，使用中证500股票池
    calculate_factors(index_code="000905.SH",
                      stock_num=10,
                      start_date='20200101',
                      end_date='20200201')
