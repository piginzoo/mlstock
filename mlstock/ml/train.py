import logging
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from mlstock.data import data_filter, data_loader
from mlstock.data.datasource import DataSource
from mlstock.data.stock_info import StocksInfo
from mlstock.factors.daily_indicator import DailyIndicator
from mlstock.factors.kdj import KDJ
from mlstock.factors.macd import MACD
from mlstock.factors.psy import PSY
from mlstock.factors.rsi import RSI
from mlstock.factors.balance_sheet import BalanceSheet
from mlstock.factors.cashflow import CashFlow
from mlstock.factors.income import Income
from mlstock.factors.std import Std
from mlstock.factors.returns import Return
from mlstock.factors.turnover_return import TurnoverReturn
from mlstock.utils import utils
from mlstock.utils.utils import time_elapse

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FACTORS = [Return, TurnoverReturn, Std, MACD, KDJ, PSY, RSI, BalanceSheet, Income, CashFlow, DailyIndicator]


def main(start_date, end_date, num):
    start_time = time.time()
    datasource = DataSource()

    # 过滤非主板、非中小板股票、且上市在1年以上的非ST股票
    df_stock_basic = data_filter.filter_stocks()
    df_stock_basic = df_stock_basic.iloc[:num]
    stocks_info = StocksInfo(df_stock_basic.ts_code, start_date, end_date)

    # 临时保存一下，用于本地下载数据提供列表（调试用）
    df_stock_basic.ts_code.to_csv("data/stocks.txt", index=False)

    # 加载周频数据
    stock_data = data_loader.load(datasource, df_stock_basic.ts_code, start_date, end_date)

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

    logger.info("因子获取完成，合计%d个因子%r，%d 行数据", len(factor_names), factor_names, len(df_weekly))

    # 因为前面的日期中，为了防止MACD之类的技术指标出现NAN预加载了数据，所以要过滤掉这些start_date之前的数据
    original_length = len(df_weekly)
    df_weekly = df_weekly[df_weekly.trade_date >= start_date]
    logger.debug("过滤掉[%s]之前的数据（为防止技术指标nan）后：%d => %d 行", start_date, original_length, len(df_weekly))

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

    """
    每一列，都去极值（TODO：是不是按照各股自己的值来做是不是更好？现在是所有的股票）
    中位数去极值:
    - 设第 T 期某因子在所有个股上的暴露度序列为𝐷𝑖
    - 𝐷𝑀为该序列中位数
    - 𝐷𝑀1为序列|𝐷𝑖 − 𝐷𝑀|的中位数
    - 则将序列𝐷𝑖中所有大于𝐷𝑀 + 5𝐷𝑀1的数重设为𝐷𝑀 + 5𝐷𝑀1
    - 将序列𝐷𝑖中所有小于𝐷𝑀 − 5𝐷𝑀1的数重设为𝐷𝑀 − 5𝐷𝑀1
    """

    def scaller(x):
        _max = df_median[x.name] + 5 * df_scope[x.name]
        _min = df_median[x.name] - 5 * df_scope[x.name]
        x = x.apply(lambda v: _min if v < _min else v)
        x = x.apply(lambda v: _max if v > _max else v)
        return x

    # 保留feature
    df_features = df_weekly[factor_names]
    # 每列都求中位数，和中位数之差的绝对值的中位数
    df_median = df_features.median()
    df_scope = df_features.apply(lambda x: x - df_median[x.name]).abs().median()
    df_features = df_features.apply(scaller)
    df_weekly[factor_names] = df_features

    # 标准化
    scaler = StandardScaler()
    scaler.fit(df_weekly[factor_names])
    df_weekly[factor_names] = scaler.transform(df_weekly[factor_names])
    logger.info("对%d个特征进行了标准化(中位数去极值)处理：%d 行", len(factor_names), len(df_weekly))

    # 去除所有的NAN数据
    logger.info("NA统计：数据特征中的NAN数：\n%r", df_weekly[factor_names].isna().sum())
    df_weekly = filter_invalid_data(df_weekly, factor_names)

    df_weekly.dropna(subset=factor_names + ['target'], inplace=True)
    logger.info("去除NAN后，数据剩余行数：%d 行", len(df_weekly))

    df_data = df_weekly[['ts_code', 'trade_date'] + factor_names + ['target']]
    csv_file_name = "data/{}_{}_{}.csv".format(start_date, end_date, utils.now())
    df_data.to_csv(csv_file_name, index=False)
    logger.info("保存 %d 行（训练和测试）数据到文件：%s", len(df_data), csv_file_name)
    start_time = time_elapse(start_time, "加载数据和清洗特征")

    # 准备训练用数据，需要numpy类型
    assert len(df_weekly) > 0
    X_train = df_weekly[factor_names].values
    y_train = df_weekly.target

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
    plt.figure(figsize=(20, 5))
    plt.title('Best Apha')
    plt.plot(alpha_scope, results, c="red", label="alpha")
    plt.legend()
    plt.show()

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


# python -m mlstock.ml.train
if __name__ == '__main__':
    utils.init_logger(file=False, log_level=logging.INFO)
    start_date = "20180101"
    end_date = "20220101"
    num = 20
    main(start_date, end_date, num)
