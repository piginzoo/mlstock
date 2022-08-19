import argparse
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

from mlstock.data import data_filter, data_loader
from mlstock.data.datasource import DataSource
from mlstock.data.stock_info import StocksInfo
from mlstock.ml.factor_conf import FACTORS
from mlstock.utils import utils
from mlstock.utils.utils import time_elapse

logger = logging.getLogger(__name__)


def load(start_date, end_date, num):
    start_time = time.time()
    datasource = DataSource()

    # 过滤非主板、非中小板股票、且上市在1年以上的非ST股票
    df_stock_basic = data_filter.filter_stocks()
    df_stock_basic = df_stock_basic.iloc[:num]
    ts_codes = df_stock_basic.ts_code
    # ts_codes = ['000401.SZ']
    stocks_info = StocksInfo(ts_codes, start_date, end_date)

    # 临时保存一下，用于本地下载数据提供列表（调试用）
    # df_stock_basic.ts_code.to_csv("data/stocks.txt", index=False)

    # 加载周频数据
    stock_data = data_loader.load(datasource, ts_codes, start_date, end_date)

    # 把基础信息merge到周频数据中
    df_weekly = stock_data.df_weekly.merge(df_stock_basic, on='ts_code', how='left')

    # 某只股票上市12周内的数据扔掉，不需要
    old_length = len(df_weekly)
    a = pd.to_datetime(df_weekly.trade_date, format='%Y%m%d')
    b = pd.to_datetime(df_weekly.list_date, format='%Y%m%d')
    df_weekly = df_weekly[a - b > pd.Timedelta(12, unit='w')]
    logger.info("剔除掉上市12周内的数据：%d=>%d", old_length, len(df_weekly))

    factor_names = []
    # 获取每一个因子（特征），并且，并入到股票数据中
    for factor_class in FACTORS:
        factor = factor_class(datasource, stocks_info)
        df_factor = factor.calculate(stock_data)
        df_weekly = factor.merge(df_weekly, df_factor)
        factor_names += factor.name if type(factor.name) == list else [factor.name]
        logger.info("获取因子%r %d 行数据", factor.name, len(df_factor))

    logger.info("因子加载完成，合计 %d 行数据，%d个因子:\n%r", len(df_weekly), len(factor_names), factor_names)
    time_elapse(start_time, "⭐️ 全部因子加载")

    df_weekly = load_index(df_weekly, start_date, end_date, datasource)

    save_csv("raw", df_weekly, start_date, end_date)

    return df_weekly, factor_names


def load_index(df_weekly, start_date, end_date, datasource):
    """
    加载基准的收益率
    :param df_weekly:
    :param datasource:
    :return:
    """

    # 合并沪深300的周收益率，为何用它呢，是为了计算超额收益(r_i = pct_chg - pct_chg_hs300)
    df_hs300 = datasource.index_weekly("000300.SH", start_date, end_date)
    df_hs300 = df_hs300[['trade_date', 'pct_chg']]
    df_hs300 = df_hs300.rename(columns={'pct_chg': 'pct_chg_hs300'})
    logger.info("下载沪深300 %s~%s 数据 %d 条", start_date, end_date, len(df_hs300))

    df_weekly = df_weekly.merge(df_hs300, on=['trade_date'], how='left')
    logger.info("合并沪深300 %d=>%d", len(df_weekly), len(df_weekly))

    # 计算出和基准（沪深300）的超额收益率，并且基于它，设置预测标签'target'（预测下一期，所以做shift）
    df_weekly['rm_rf'] = df_weekly.pct_chg - df_weekly.pct_chg_hs300

    # target即预测目标，是下一期的超额收益
    df_weekly['target'] = df_weekly.groupby('ts_code').rm_rf.shift(-1)
    return df_weekly


def _scaller(x, df_median, df_scope):
    """
    - 则将序列𝐷𝑖中所有大于𝐷𝑀 + 5𝐷𝑀1的数重设为𝐷𝑀 + 5𝐷𝑀1
    - 将序列𝐷𝑖中所有小于𝐷𝑀 − 5𝐷𝑀1的数重设为𝐷𝑀 − 5𝐷𝑀1
    :param x: 就是某一列，比如beta
        Name: beta, Length: 585, dtype: float64
        180          NaN
        181          NaN
                  ...
        1196    163121.0
    :param df_median:
        (Pdb) df_median
        return_1w                 -0.002050
        return_3w                 -0.007407
        .....                     ......
        alpha                     0.000161
        beta                      0.276572
        stake_holder              163121.000000
        Length: 73, dtype: float64
    :param df_scope:
        (Pdb) df_scope
        return_1w                 0.029447
        .....                     ......
        stake_holder              82657.000000
        Length: 73, dtype: float64
    :return:
    """
    _max = df_median[x.name] + 5 * df_scope[x.name]
    _min = df_median[x.name] - 5 * df_scope[x.name]
    x = x.apply(lambda v: _min if v < _min else v)
    x = x.apply(lambda v: _max if v > _max else v)
    return x


def process(df_features, factor_names, start_date, end_date):
    """

    :param df_features:
    :param factor_names:
    :param start_date: 因为前面的日期中，为了防止MACD之类的技术指标出现NAN预加载了数据，所以要过滤掉这些start_date之前的数据
    :return:
    """

    """
    因为前面的日期中，为了防止MACD之类的技术指标出现NAN预加载了数据，所以要过滤掉这些start_date之前的数据
    """
    original_length = len(df_features)
    df_features = df_features[df_features.trade_date >= start_date]
    logger.info("过滤掉[%s]之前的数据（为防止技术指标nan）后：%d => %d 行", start_date, original_length, len(df_features))

    logger.info("(调试)特征处理之前的数据情况：\n%r", df_features.describe())
    logger.info("(调试)特征处理之前NA统计：数据特征中的NAN数：\n%r", df_features[factor_names].isna().sum().sort_values())

    """
    如果target缺失比较多，就删除掉这些股票
    """
    original_length = len(df_features)
    df_features = df_features[~df_features.target.isna()]
    logger.info("过滤掉target为nan的行后：%d => %d 行，剔除占比%.1f%%",
                original_length,
                len(df_features),
                (original_length - len(df_features)) * 100 / original_length)

    """
    去除那些因子值中超过20%缺失的股票（看所有因子中确实最大的那个，百分比超过20%，这只股票整个剔除掉）
    """
    # 计算每只股票的每个特征的缺失百分比
    df_na_miss_percent_by_code = df_features.groupby(by='ts_code').apply(
        lambda df: (df.shape[0] - df.count()) / df.shape[0])

    # 找出最大的那个特征的缺失比，如果其>80%，就剔除这只股票
    df_na_miss_codes = df_na_miss_percent_by_code[df_na_miss_percent_by_code.max(axis=1) > 0.8]['ts_code']
    # 把这些行找出来，打印到日志中，方便后期调试
    df_missed_info = df_na_miss_percent_by_code[
        df_na_miss_percent_by_code.apply(lambda x: x.name in df_na_miss_codes, axis=1)]
    # 0缺失的列，需要扣掉，只保留确实列打印出来调试
    need_drop_columns = df_missed_info.sum()[df_missed_info.sum() == 0].index
    df_missed_info = df_missed_info.drop(need_drop_columns, axis=1)
    logger.info("(调试)以下股票的某些特征的'缺失(NA)率'，超过80%%，%d 只(需要被删掉的股票)：\n%r", len(df_missed_info), df_missed_info)
    # 剔除这些问题股票
    origin_stock_size = len(df_features.ts_code.unique())
    origin_data_size = df_features.shape[0]
    df_features = df_features[df_features.ts_code.apply(lambda x: x not in df_na_miss_codes)]
    logger.info("从%d只股票中剔除了%d只，占比%.1f%%；剔除相关数据%d=>%d行，剔除占比%.2f%%",
                origin_stock_size,
                len(df_na_miss_codes),
                len(df_na_miss_codes) * 100 / origin_stock_size,
                origin_data_size,
                len(df_features),
                (origin_data_size - len(df_features)) * 100 / origin_data_size)

    """
    去除极值+标准化
    每一列，都去极值（TODO：是不是按照各股自己的值来做是不是更好？现在是所有的股票）
    中位数去极值:
    - 设第 T 期某因子在所有个股上的暴露度序列为𝐷𝑖
    - 𝐷𝑀为该序列中位数
    - 𝐷𝑀1为序列|𝐷𝑖 − 𝐷𝑀|的中位数
    - 则将序列𝐷𝑖中所有大于𝐷𝑀 + 5𝐷𝑀1的数重设为𝐷𝑀 + 5𝐷𝑀1
    - 将序列𝐷𝑖中所有小于𝐷𝑀 − 5𝐷𝑀1的数重设为𝐷𝑀 − 5𝐷𝑀1
    """
    # 每列都求中位数，和中位数之差的绝对值的中位数
    df_features_only = df_features[factor_names]
    # 找到每一个特征的中位值
    df_median = df_features_only.median()
    # 每个值，都和中位数相减后，取绝对值，然后在找到绝对值们的中位数，这个就是要限定的范围值
    df_scope = df_features_only.apply(lambda x: x - df_median[x.name]).abs().median()
    df_features_only = df_features_only.apply(lambda x: _scaller(x, df_median, df_scope))

    # 标准化：
    # 将中性化处理后的因子暴露度序列减去其现在的均值、除以其标准差，得到一个新的近似服从N(0,1)分布的序列。
    scaler = StandardScaler()
    scaler.fit(df_features_only)
    df_features[factor_names] = scaler.transform(df_features_only)
    logger.info("对%d个特征进行了标准化(中位数去极值)处理：%d 行", len(factor_names), len(df_features))

    # 去除所有的NAN数据
    logger.info("NA统计：数据特征中的NAN数：\n%r", df_features[factor_names].isna().sum().sort_values())
    df_features = filter_invalid_data(df_features, factor_names)

    original_length = len(df_features)
    df_features.dropna(subset=factor_names + ['target'], inplace=True)
    logger.info("去除NAN后，数据剩余行数：%d=>%d 行，剔除了%.1f%%",
                original_length,
                len(df_features),
                (original_length - len(df_features)) * 100 / original_length)

    """
    去重
    """
    original_length = len(df_features)
    df_features = df_features[~df_features[['ts_code', 'trade_date']].duplicated()].reset_index(drop=True)
    logger.info("去除重复行(ts_code+trade_date)后，数据 %d => %d 行，剔除了%.1f%%",
                original_length,
                len(df_features),
                (original_length - len(df_features)) * 100 / original_length)

    save_csv("features", df_features, start_date, end_date)

    logger.info("特征处理之后的数据情况：\n%r", df_features.describe())

    return df_features


def save_csv(name, df, start_date, end_date):
    csv_file_name = "data/{}_{}_{}_{}.csv".format(name, start_date, end_date, utils.now())
    df.to_csv(csv_file_name, index=False)
    logger.info("保存 %d 行数据到文件：%s", len(df), csv_file_name)


def main(start_date, end_date, num):
    # 加载特征、基准收益
    df_weekly, factor_names = load(start_date, end_date, num)

    # 处理特征，剔除异常等
    df_features = df_weekly[['ts_code', 'trade_date', 'target'] + factor_names]
    df_features = process(df_features, factor_names, start_date, end_date)

    # 准备训练用数据，需要numpy类型
    assert len(df_weekly) > 0
    X_train = df_features[factor_names].values
    y_train = df_features.target

    # 划分训练集和测试集，测试集占总数据的15%，随机种子为10
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15, random_state=10)

    # 使用交叉验证，分成10份，挨个做K-Fold，训练
    cv_scores = []
    for n in range(5):
        regession = linear_model.LinearRegression()
        scores = cross_val_score(regession, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        cv_scores.append(scores.mean())
    logger.info("成绩：\n%r", cv_scores)

    # 做这个是为了人肉看一下最好的岭回归的超参alpha的最优值是啥
    # 是没必要的，因为后面还会用 gridsearch自动跑一下，做这个就是想直观的感受一下
    results = []
    alpha_scope = np.arange(200, 500, 5)
    for i in alpha_scope:
        ridge = Ridge(alpha=i)
        results.append(cross_val_score(ridge, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean())
    logger.info("最好的参数：%.0f, 对应的最好的均方误差：%.2f",
                alpha_scope[results.index(max(results))],
                max(results))
    fig = plt.figure(figsize=(20, 5))
    plt.title('Best Alpha')
    plt.plot(alpha_scope, results, c="red", label="alpha")
    plt.legend()
    fig.savefig("data/best_alpha.jpg")

    # 用grid search找最好的alpha：[200,205,...,500]
    # grid search的参数是alpha，岭回归就这样一个参数，用于约束参数的平方和
    # grid search的入参包括alpha的范围，K-Fold的折数(cv)，还有岭回归评价的函数(负均方误差)
    grid_search = GridSearchCV(Ridge(),
                               {'alpha': alpha_scope},
                               cv=5,  # 5折(KFold值)
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    # model = grid_search.estimator.fit(X_train, y_train)
    logger.info("GridSarch最好的成绩:%.5f", grid_search.best_score_)
    # 得到的结果是495，确实和上面人肉跑是一样的结果
    logger.info("GridSarch最好的参数:%.5f", grid_search.best_estimator_.alpha)


def filter_invalid_data(df, factor_names):
    for factor_name in factor_names:
        original_size = len(df)
        # 去掉那些这个特征全是nan的股票
        valid_ts_codes = df.groupby('ts_code')[factor_name].count()[lambda x: x > 0].index
        df = df[df['ts_code'].isin(valid_ts_codes)]
        if len(df) != original_size:
            logger.info("去除特征[%s]全部为Nan的股票数据后，行数变化：%d => %d",
                        factor_name, original_size, len(df))
    return df


"""
python -m mlstock.ml.train
python -m mlstock.ml.train -n 10 -d
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start_date', type=str, default="20090101", help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, default="20220801", help="结束日期")
    parser.add_argument('-n', '--num', type=int, default=100000, help="股票数量，调试用")
    parser.add_argument('-d', '--debug', action='store_true', default=False, help="是否调试")
    args = parser.parse_args()

    if args.debug:
        print("【调试模式】")
        utils.init_logger(file=True, log_level=logging.DEBUG)
    else:
        utils.init_logger(file=True, log_level=logging.INFO)

    main(args.start_date, args.end_date, args.num)
