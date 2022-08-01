import logging
import time

import pandas as pd

from mlstock import const
from mlstock.data.stock_data import StockData
from mlstock.utils import utils
from mlstock.utils.utils import logging_time

logger = logging.getLogger(__name__)


@logging_time('加载日频、周频、基础数据')
def load(datasource, stock_codes, start_date, end_date):
    start_time = time.time()

    # 多加载之前的数据，这样做是为了尽量不让技术指标，如MACD之类的出现NAN
    start_date = utils.last_week(start_date, const.RESERVED_PERIODS)

    # 加载周频数据
    df_weekly = __load(stock_codes, start_date, end_date, func=datasource.weekly)
    logger.info("加载[%d]只股票 %s~%s 的周频数据 %d 行，耗时%.0f秒",
                len(stock_codes),
                start_date,
                end_date,
                len(df_weekly),
                time.time() - start_time)

    start_time = time.time()
    # 加日周频数据，虽然我们算的周频，但是有些地方需要日频数据
    df_daily = __load(stock_codes, start_date, end_date, func=datasource.daily)
    logger.info("加载[%d]只股票 %s~%s 的日频数据 %d 行，耗时%.0f秒",
                len(stock_codes),
                start_date,
                end_date,
                len(df_daily),
                time.time() - start_time)

    start_time = time.time()
    # 加日周频数据，虽然我们算的周频，但是有些地方需要日频数据
    df_daily_basic = __load(stock_codes, start_date, end_date, func=datasource.daily_basic)
    logger.info("加载[%d]只股票 %s~%s 的日频基础(basic)数据 %d 行，耗时%.0f秒",
                len(stock_codes),
                start_date,
                end_date,
                len(df_daily),
                time.time() - start_time)

    stock_data = StockData()
    stock_data.df_daily = df_daily
    stock_data.df_weekly = df_weekly
    stock_data.df_daily_basic = df_daily_basic
    return stock_data


def __load(stocks, start_date, end_date, func):
    data_list = []
    for code in stocks:
        df = func(code, start_date, end_date)
        data_list.append(df)
    return pd.concat(data_list)
