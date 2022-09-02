from datetime import datetime
import logging
import numpy as np
from empyrical import max_drawdown

from mlstock.const import RISK_FREE_ANNUALLY_RETRUN
from mlstock.utils import utils

logger = logging.getLogger(__name__)


def scope(df):
    start_date = datetime.strftime(df.trade_date.min(), "YYYYMMDD")
    end_date = datetime.strftime(df.trade_date.max(), "YYYYMMDD")
    years = end_date.year - start_date.year
    months = years * 12 + end_date.month - start_date.month
    weeks, _ = divmod((end_date - start_date).days, 7)
    return f"{years}年{months}月{weeks}周"


def annually_profit(df):
    """
    年化收益率
    A股每年250个交易日，50个交易周，
    年化收益率 =
    """
    # 累计收益
    cumulative_return = df['cumulative_pct_chg'] + 1
    total_weeks = len(df)
    return np.power(cumulative_return, 50 / total_weeks) - 1


def volatility(df):
    """波动率"""
    return df['next_pct_chg'].std()


def sharp_ratio(df):
    """
    夏普比率 = 收益均值-无风险收益率 / 收益方差
    无风险收益率,在我国无风险收益率一般取值十年期国债收益
    """
    return (df['next_pct_chg'].mean() - RISK_FREE_ANNUALLY_RETRUN / 50) / df['next_pct_chg'].mean()


def max_drawback(df):
    """最大回撤"""
    return max_drawdown(df.next_pct_chg)


def annually_active_return(df):
    """年化主动收益率"""
    cumulative_active_return = df['cumulative_active_pct_chg'] + 1
    total_weeks = len(df)
    return np.power(cumulative_active_return, 50 / total_weeks) - 1


def active_return_max_drawback(df):
    """年化主动收最大回撤"""
    pass


def annually_track_error(df):
    """年化跟踪误差"""


def information_ratio(df):
    """
    信息比率IR
    IR= IC的多周期均值/IC的标准方差，代表因子获取稳定Alpha的能力。当IR大于0.5时因子稳定获取超额收益能力较强。
    - https://www.zhihu.com/question/342944058
    - https://zhuanlan.zhihu.com/p/351462926
    讲人话：
    就是主动收益的均值/主动收益的方差
    """
    return df.active_pct_chg.mean() / df.active_pct_chg.std()


def win_rate(df):
    """胜率"""
    return (df['active_pct_chg'] > 0).sum() / len(df)


def metrics(df):
    """
    :param df:
        df[
            'trade_date',
            'next_pct_chg',                 # 当前收益率
            'next_pct_chg_baseline',        # 当期基准收益率
            'cumulative_pct_chg',           # 当期累计收益率
            'cumulative_pct_chg_baseline'   # 当期基准收益率
        ]
    :param df_base:
    :return:
    """
    if df is not None:
        df.to_csv("data/df_portfolio.csv")
    else:
        import pandas as pd
        df = pd.read_csv("data/df_portfolio.csv")
        df['trade_date'] = df['trade_date'].astype(str)

    # 每期超额收益率
    df['active_pct_chg'] = df['next_pct_chg'] - df['next_pct_chg_baseline']
    # 每期累计超额收益率
    df['cumulative_active_pct_chg'] = df['cumulative_pct_chg'] - df['cumulative_pct_chg_baseline']

    result = {}

    result['投资时长'] = scope(df)
    result['累计收益率'] = df['cumulative_pct_chg'][-1]
    result['累计基准收益率'] = df['cumulative_pct_chg_baseline'][-1]
    result['年化收益率'] = annually_profit(df)
    result['年化超额收益率'] = annually_active_return(df)
    result['波动率'] = volatility(df)
    result['夏普比率'] = sharp_ratio(df)
    result['最大回撤'] = max_drawback(df)
    result['信息比率'] = information_ratio(df)
    result['胜率'] = win_rate(df)

    logger.info("投资详细指标：")
    for k, v in result.items():
        if type(v) == float and v < 1:
            v = "{:1f}%".format(v * 100)
        logger.debug("\t%s\t : %s", k, v)
    return result


# python -m mlstock.ml.evaluate.metrics
if __name__ == '__main__':
    utils.init(file=False)
    metrics(None)
