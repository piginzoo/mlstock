import argparse
import logging
import time

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pandas import DataFrame

from mlstock.const import TOP_30
from mlstock.data.datasource import DataSource
from mlstock.ml import load_and_filter_data
from mlstock.ml.backtests import backtest_simple, backtest_backtrader
from mlstock.ml.data import factor_conf
from mlstock.ml.backtests.metrics import metrics
from mlstock.utils import utils

logger = logging.getLogger(__name__)


def main(type, data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names):
    """
    回测
    :param data_path: 因子数据文件的路径
    :param start_date: 回测的开始日期
    :param end_date: 回测结束日期
    :param model_pct_path: 回测用的预测收益率的模型路径
    :param model_winloss_path: 回测用的，预测收益率的模型路径
    :param factor_names: 因子们的名称，用于过滤预测的X
    :return:
    """
    if type == 'simple':
        backtest_simple.main(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names)
    else:
        backtest_backtrader.main(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names)


"""
python -m mlstock.ml.backtest \
-s 20190101 -e 20220901 \
-mp model/pct_ridge_20220828190251.model \
-mw model/winloss_xgboost_20220828190259.model \
-d data/processed_industry_neutral_20080101_20220901_20220828152110.csv

python -m mlstock.ml.backtest \
-s 20190101 -e 20220901 \
-mp model/pct_ridge_20220828190251.model \
-mw model/winloss_xgboost_20220828190259.model \
-d data/factor__20080101_20220901_50_30679__industry_neutral_20220902095251.csv
"""
if __name__ == '__main__':
    utils.init_logger(file=True)
    parser = argparse.ArgumentParser()

    # 数据相关的
    parser.add_argument('-t', '--type', type=str, default="simple")
    parser.add_argument('-s', '--start_date', type=str, default="20190101", help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, default="20220901", help="结束日期")
    parser.add_argument('-d', '--data', type=str, default=None, help="数据文件")
    parser.add_argument('-mp', '--model_pct', type=str, default=None, help="收益率模型")
    parser.add_argument('-mw', '--model_winloss', type=str, default=None, help="涨跌模型")

    args = parser.parse_args()

    factor_names = factor_conf.get_factor_names()

    main(
        args.type,
        args.data,
        args.start_date,
        args.end_date,
        args.model_pct,
        args.model_winloss,
        factor_names)
