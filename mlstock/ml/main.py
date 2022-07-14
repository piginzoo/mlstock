import time

from mlstock.data import data_filter, data_loader
from mlstock.data.datasource import DataSource
from mlstock.factors.KDJ import KDJ
from mlstock.factors.MACD import MACD
from mlstock.utils import utils
import logging
import pandas as pd
import numpy as np
import math

logger = logging.getLogger(__name__)

FACTORS = [MACD, KDJ]


def main(start_date, end_date, num):
    start_time = time.time()
    datasource = DataSource()
    df_stocks = data_filter.filter_stocks()
    df_stocks = df_stocks[:num]
    df_stocks_data = data_loader.weekly(datasource, df_stocks.ts_code, start_date, end_date)
    df_stocks = df_stocks.merge(df_stocks_data, on=['ts_code'], how="left")
    logger.debug("加载[%d]只股票 %s~%s 的数据 %d 行，耗时%.0f秒", len(df_stocks), start_date, end_date, len(df_stocks),
                 time.time() - start_time)

    for factor_class in FACTORS:
        factor = factor_class(datasource)
        seris_factor = factor.calculate(df_stocks)
        df_stocks[factor.name] = seris_factor

    df_hs300 = datasource.index_weekly("000300.SH", start_date, end_date)
    df_hs300 = df_hs300[['trade_date', 'pct_chg']]
    df_hs300 = df_hs300.rename(columns={'pct_chg': 'pct_chg_hs300'})
    logger.debug("下载沪深300 %s~%s 数据 %d 条", start_date, end_date, len(df_hs300))

    df_stocks = df_stocks.merge(df_hs300, on=['trade_date'], how='left')
    logger.debug("合并沪深300 %d=>%d", len(df_stocks), len(df_stocks))

    df_stocks['rm_rf'] = df_stocks.pct_chg - df_stocks.pct_chg_hs300
    df_stocks['target'] = df_stocks.groupby('ts_code').rm_rf.shift(-1)

    # 按照0.8:0.2和时间顺序，划分train和test
    trade_dates = df_stocks.trade_date.sort_values().unique()
    div_num = math.ceil(len(trade_dates) * 0.8)
    train_dates = trade_dates[:div_num]
    test_dates = trade_dates[div_num:]
    df_train = df_stocks[df_stocks.trade_date.apply(lambda x: x in train_dates)]
    df_test = df_stocks[df_stocks.trade_date.apply(lambda x: x in test_dates)]

    # 某只股票上市12周内的数据扔掉，不需要
    a = pd.to_datetime(df_train.trade_date, format='%Y%m%d')
    b = pd.to_datetime(df_train.list_date, format='%Y%m%d')
    df_train = df_train[a - b > pd.Timedelta(12, unit='w')]

    a = pd.to_datetime(df_test.trade_date, format='%Y%m%d')
    b = pd.to_datetime(df_test.list_date, format='%Y%m%d')
    df_test = df_test[a - b > pd.Timedelta(12, unit='w')]

    df_train.target.count() / df_train.shape[0]

    """
    每一列，都去极值（TODO：是不是按照各股自己的值来做是不是更好？现在是所有的股票）
    中位数去极值:
    - 设第 T 期某因子在所有个股上的暴露度序列为𝐷𝑖
    - 𝐷𝑀为该序列中位数
    - 𝐷𝑀1为序列|𝐷𝑖 − 𝐷𝑀|的中位数
    - 则将序列𝐷𝑖中所有大于𝐷𝑀 + 5𝐷𝑀1的数重设为𝐷𝑀 + 5𝐷𝑀1
    - 将序列𝐷𝑖中所有小于𝐷𝑀 − 5𝐷𝑀1的数重设为𝐷𝑀 − 5𝐷𝑀1
    """
    # 保留feature
    feature_names = ['MACD', 'KDJ']
    df_features = df_train[feature_names]
    # 每列都求中位数，和中位数之差的绝对值的中位数
    df_median = df_features.median()
    df_scope = df_features.apply(lambda x: x - df_median[x.name]).abs().median()
    df_scope

    def scaller(x):
        _max = df_median[x.name] + 5 * df_scope[x.name]
        _min = df_median[x.name] - 5 * df_scope[x.name]
        x = x.apply(lambda v: _min if v < _min else v)
        x = x.apply(lambda v: _max if v > _max else v)
        return x

    df_features = df_features.apply(scaller)
    df_train[feature_names] = df_features

    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_train[feature_names])
    df_train[feature_names] = scaler.transform(df_train[feature_names])
    df_test[feature_names] = scaler.transform(df_test[feature_names])

    from sklearn import linear_model

    df_train = df_train[feature_names + ['target']]
    df_train.dropna(inplace=True)
    df_test = df_train[feature_names + ['target']]
    df_test.dropna(inplace=True)

    X_train = df_train[feature_names].values
    X_test = df_test[feature_names].values
    y_train = df_train.target
    y_test = df_test.target

    reg = linear_model.LinearRegression()
    print(X_train.shape, y_train.shape)
    model = reg.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("y_pred", y_pred)


# python -m mlstock.ml.main
if __name__ == '__main__':
    utils.init_logger(simple=True)
    start_date = "20180101"
    end_date = "20220101"
    num = 20
    main(start_date, end_date, num)
