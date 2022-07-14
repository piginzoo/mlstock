import logging
import pandas as pd

logger = logging.getLogger(__name__)


def __load(stocks, start_date, end_date, func):
    data_list = []
    for code in stocks:
        df = func(code, start_date, end_date)
        data_list.append(df)
    return pd.concat(data_list)


def daily(datasource, stocks, start_date, end_date):
    return __load(stocks, start_date, end_date, func=datasource.daily)


def weekly(datasource, stocks, start_date, end_date):
    return __load(stocks, start_date, end_date, func=datasource.weekly)


def monthly(datasource, stocks, start_date, end_date):
    return __load(stocks, start_date, end_date, func=datasource.monthly)
