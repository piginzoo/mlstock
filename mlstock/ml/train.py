import logging
import math
import time
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from mlstock.data import data_filter, data_loader
from mlstock.data.datasource import DataSource
from mlstock.factors.KDJ import KDJ
from mlstock.factors.MACD import MACD
from mlstock.factors.balance_sheet import BalanceSheet
from mlstock.utils import utils
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FACTORS = [MACD, KDJ, BalanceSheet]


class StocksInfo:
    def __init__(self, stocks, start_date, end_date):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date


def main(start_date, end_date, num):
    start_time = time.time()
    datasource = DataSource()
    stocks = data_filter.filter_stocks()
    stocks = stocks[:num]

    stocks_info = StocksInfo(stocks, start_date, end_date)

    df_stocks_data = data_loader.weekly(datasource, stocks.ts_code, start_date, end_date)
    df_stocks = stocks.merge(df_stocks_data, on=['ts_code'], how="left")
    logger.debug("加载[%d]只股票 %s~%s 的数据 %d 行，耗时%.0f秒", len(df_stocks), start_date, end_date, len(df_stocks),
                 time.time() - start_time)

    df_factors = []
    factor_names = []
    for factor_class in FACTORS:
        factor = factor_class(datasource, stocks_info)
        df_factors.append(factor.calculate(df_stocks))
        factor_names.append(factor.name)

    utils.fill

    # 合并沪深300的周收益率，为何用它呢，是为了计算超额收益(r_i = pct_chg - pct_chg_hs300)
    df_hs300 = datasource.index_weekly("000300.SH", start_date, end_date)
    df_hs300 = df_hs300[['trade_date', 'pct_chg']]
    df_hs300 = df_hs300.rename(columns={'pct_chg': 'pct_chg_hs300'})
    logger.debug("下载沪深300 %s~%s 数据 %d 条", start_date, end_date, len(df_hs300))
    df_stocks = df_stocks.merge(df_hs300, on=['trade_date'], how='left')
    logger.debug("合并沪深300 %d=>%d", len(df_stocks), len(df_stocks))
    # 计算出和基准（沪深300）的超额收益率，并且基于它，设置预测标签'target'（预测下一期，所以做shift）
    df_stocks['rm_rf'] = df_stocks.pct_chg - df_stocks.pct_chg_hs300
    # target即预测目标，是下一期的超额收益
    df_stocks['target'] = df_stocks.groupby('ts_code').rm_rf.shift(-1)

    # 某只股票上市12周内的数据扔掉，不需要
    df_train = df_stocks
    a = pd.to_datetime(df_train.trade_date, format='%Y%m%d')
    b = pd.to_datetime(df_train.list_date, format='%Y%m%d')
    df_train = df_train[a - b > pd.Timedelta(12, unit='w')]

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

    # 去除所有的NAN数据
    df_train.dropna(subset=feature_names + ['target'], inplace=True)
    logger.debug("NA统计：train data：%r", df_train[feature_names].isna().sum())

    # 准备训练用数据，需要numpy类型
    X_train = df_train[feature_names].values
    y_train = df_train.target

    # 划分训练集和测试集，测试集占总数据的15%，随机种子为10
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15, random_state=10)

    # 使用交叉验证，分成10份，挨个做K-Fold，训练
    cv_scores = []
    for n in range(5):
        regession = linear_model.LinearRegression()
        scores = cross_val_score(regession, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        cv_scores.append(scores.mean())
    logger.debug("成绩：\n%r", cv_scores)

    # 做这个是为了人肉看一下最好的岭回归的超参alpha的最优值是啥
    # 是没必要的，因为后面还会用 gridsearch自动跑一下，做这个就是想直观的感受一下
    results = []
    alpha_scope = np.arange(200, 500, 5)
    for i in alpha_scope:
        ridge = Ridge(alpha=i)
        results.append(cross_val_score(ridge, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean())
    logger.debug("最好的参数：%.0f, 对应的最好的均方误差：%.2f",
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
    logger.debug("GridSarch最好的成绩:%.5f", grid_search.best_score_)
    # 得到的结果是495，确实和上面人肉跑是一样的结果
    logger.debug("GridSarch最好的参数:%.5f", grid_search.best_estimator_.alpha)


# python -m mlstock.ml.train
if __name__ == '__main__':
    utils.init_logger(simple=True)
    start_date = "20180101"
    end_date = "20220101"
    num = 20
    main(start_date, end_date, num)
