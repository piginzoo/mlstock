import logging
import time

import pandas as pd
import numpy as np
from mlstock import const
from mlstock.data.stock_data import StockData
from mlstock.utils import utils
from mlstock.utils.utils import logging_time

logger = logging.getLogger(__name__)


def calculate_columns_missed_by_stock(df, columns):
    """
    按照股票代码，来算他的所有的特征中，NA的占比
    返回的是一个包含了index为ts_code的NA占比值：
    ts_code
    000505.SZ    0.081690
    000506.SZ    0.532946
    000507.SZ    0.000000
    000509.SZ    0.534615
    000510.SZ    0.000000
    """
    assert 'ts_code' in df.columns

    # 如果有na，count会不计数，利用这个特性来计算缺失比例
    return (1 - df[columns].groupby('ts_code').apply(lambda d: d.count() / d.shape[0])).max(axis=1)


@logging_time('加载日频、周频、基础数据')
def load(datasource, stock_codes, start_date, end_date):
    """从数据库加载数据，并做一些必要填充"""

    # 多加载之前的数据，这样做是为了尽量不让技术指标，如MACD之类的出现NAN
    original_start_date = start_date
    start_date = utils.last_week(start_date, const.RESERVED_PERIODS)
    logger.debug("开始加载 %s ~ %s 的股票数据（从真正开始日期%s预加载%d周）",
                 start_date, end_date, original_start_date, const.RESERVED_PERIODS)

    # 加日周频数据，虽然我们算的周频，但是有些地方需要日频数据
    start_time = time.time()
    df_daily_basic = __load(stock_codes, start_date, end_date, func=datasource.daily_basic)
    # 把daily_basic中关键字段缺少比较多（>80%）的股票剔除掉
    df_stock_nan_stat = calculate_columns_missed_by_stock(df_daily_basic,
                                                          ['ts_code', 'trade_date', 'total_mv', 'pe_ttm', 'ps_ttm',
                                                           'pb'])
    nan_too_many_stocks = df_stock_nan_stat[df_stock_nan_stat > 0.8].index
    if len(nan_too_many_stocks) > 0:
        stock_codes = stock_codes[~stock_codes.isin(nan_too_many_stocks.tolist())]
        df_daily_basic = df_daily_basic[~df_daily_basic.isin(nan_too_many_stocks.tolist())]
        logger.warning("由于daily_basic中的'total_mv','pe_ttm', 'ps_ttm', 'pb'缺失值超过80%%，导致%d只股票被剔除：%r",
                       len(nan_too_many_stocks.tolist()),
                       nan_too_many_stocks.tolist())
    # 把daily_basic的nan信息都fill上
    df_daily_basic = df_daily_basic.sort_values(['ts_code', 'trade_date'])
    df_daily_basic[['total_mv', 'pe_ttm', 'ps_ttm', 'pb']] = \
        df_daily_basic.groupby('ts_code').ffill().bfill()[['total_mv', 'pe_ttm', 'ps_ttm', 'pb']]
    # 对市值去对数，降低这个值的范围，这个新对数市值，后面行业中性化会用
    df_daily_basic['total_mv_log'] = df_daily_basic.total_mv.apply(np.log)

    logger.info("加载[%d]只股票 %s~%s 的日频基础(basic)数据 %d 行，耗时%.0f秒",
                len(stock_codes),
                start_date,
                end_date,
                len(df_daily_basic),
                time.time() - start_time)

    # 加载周频数据
    df_weekly = __load(stock_codes, start_date, end_date, func=datasource.weekly)
    logger.info("加载[%d]只股票 %s~%s 的周频数据 %d 行，耗时%.0f秒",
                len(stock_codes),
                start_date,
                end_date,
                len(df_weekly),
                time.time() - start_time)

    # 加日周频数据，虽然我们算的周频，但是有些地方需要日频数据
    start_time = time.time()
    df_daily = __load(stock_codes, start_date, end_date, func=datasource.daily)
    logger.info("加载[%d]只股票 %s~%s 的日频数据 %d 行，耗时%.0f秒",
                len(stock_codes),
                start_date,
                end_date,
                len(df_daily),
                time.time() - start_time)

    # 加上证指数的日频数据
    start_time = time.time()
    df_index_daily = __load(['000001.SH'], start_date, end_date, func=datasource.index_daily)
    logger.info("加载上证指数 %s~%s 的日频数据 %d 行，耗时%.0f秒",
                start_date,
                end_date,
                len(df_index_daily),
                time.time() - start_time)

    # 加上证指数的周频数据
    start_time = time.time()
    df_index_weekly = __load(['000001.SH'], start_date, end_date, func=datasource.index_weekly)
    logger.info("加载上证指数 %s~%s 的周频数据 %d 行，耗时%.0f秒",
                start_date,
                end_date,
                len(df_index_weekly),
                time.time() - start_time)

    # 加上交易日历数据
    df_calendar = datasource.trade_cal(start_date, end_date)

    stock_data = StockData()
    # 按照ts_code + trade_date，排序
    # 排序默认是ascending=True, 升序，从旧到新，比如日期是2008->2022，
    # 然后赋值到stock_data
    stock_data.df_daily = df_daily.sort_values(['ts_code', 'trade_date'])
    stock_data.df_weekly = df_weekly.sort_values(['ts_code', 'trade_date'])
    stock_data.df_daily_basic = df_daily_basic  # 之前sort过了
    stock_data.df_index_weekly = df_index_weekly.sort_values(['ts_code', 'trade_date'])
    stock_data.df_index_daily = df_index_daily.sort_values(['ts_code', 'trade_date'])
    stock_data.df_calendar = df_calendar

    return stock_data


def __load(stocks, start_date, end_date, func):
    data_list = []
    for code in stocks:
        df = func(code, start_date, end_date)
        data_list.append(df)
    return pd.concat(data_list)
