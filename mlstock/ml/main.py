import logging
import math
import time

import pandas as pd
from sklearn import linear_model

from mlstock.data import data_filter, data_loader
from mlstock.data.datasource import DataSource
from mlstock.factors.KDJ import KDJ
from mlstock.factors.MACD import MACD
from mlstock.utils import utils
from sklearn.preprocessing import StandardScaler

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

    # 合并沪深300的周收益率
    df_hs300 = datasource.index_weekly("000300.SH", start_date, end_date)
    df_hs300 = df_hs300[['trade_date', 'pct_chg']]
    df_hs300 = df_hs300.rename(columns={'pct_chg': 'pct_chg_hs300'})
    logger.debug("下载沪深300 %s~%s 数据 %d 条", start_date, end_date, len(df_hs300))
    df_stocks = df_stocks.merge(df_hs300, on=['trade_date'], how='left')
    logger.debug("合并沪深300 %d=>%d", len(df_stocks), len(df_stocks))

    # 计算出和基准（沪深300）的超额收益率，并且基于它，设置预测标签'target'（预测下一期，所以做shift）
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

    def scaller(x):
        _max = df_median[x.name] + 5 * df_scope[x.name]
        _min = df_median[x.name] - 5 * df_scope[x.name]
        x = x.apply(lambda v: _min if v < _min else v)
        x = x.apply(lambda v: _max if v > _max else v)
        return x

    df_features = df_features.apply(scaller)
    df_train[feature_names] = df_features

    # 标准化
    scaler = StandardScaler()
    scaler.fit(df_train[feature_names])
    df_train[feature_names] = scaler.transform(df_train[feature_names])
    df_test[feature_names] = scaler.transform(df_test[feature_names])

    # 去除所有的NAN数据
    df_train.dropna(subset=feature_names+['target'], inplace=True)
    df_test.dropna(subset=feature_names+['target'], inplace=True)
    logger.debug("NA统计：train data：%r,label：%r",
                 df_train[feature_names].isna().sum(),
                 df_test[feature_names].isna().sum())

    # 准备训练用数据，需要numpy类型
    X_train = df_train[feature_names].values
    X_test = df_test[feature_names].values
    y_train = df_train.target
    y_test = df_test.target

    # 训练
    regession = linear_model.LinearRegression()
    model = regession.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 模型评价
    df_result = pd.DataFrame({'ts_code': df_test.ts_code,
                              'trade_date': df_test.trade_date,
                              'y': y_test,
                              'y_pred': y_pred})

    # IC
    ic = df_result[['y', 'y_pred']].corr().iloc[0, 1]
    logger.info("预测值和标签的相关性(IC): %.2f%%", ic * 100)

    # Rank IC
    df_result['y_rank'] = df_result.y.rank(ascending=False)  # 并列的默认使用排名均值
    df_result['y_pred_rank'] = df_result.y_pred.rank(ascending=False)
    rank_ic = df_result[['y_rank', 'y_pred_rank']].corr().iloc[0, 1]
    logger.info("预测值和标签的排名相关性(Rank IC): %.2f%%", rank_ic * 100)

    # 分层回测，每个行业内分5类
    df_result['industry'] = df_test.industry
    df_result['y_rank_in_industry'] = df_result.groupby('industry').y_pred.rank(ascending=False)  # 每行业内排名（按行业分组）
    df_result['class_label_in_industry'] = pd.qcut(df_result.y_rank_in_industry, q=5, labels=[1, 2, 3, 4, 5],
                                                   duplicates='drop')
    print(df_result)
    print(df_result.groupby('class_label_in_industry').mean())


# python -m mlstock.ml.main
if __name__ == '__main__':
    utils.init_logger(simple=True)
    start_date = "20180101"
    end_date = "20220101"
    num = 20
    main(start_date, end_date, num)
