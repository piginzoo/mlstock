import argparse
import logging
import os.path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from mlstock.const import CODE_DATE
from mlstock.ml import data_processor
from mlstock.utils import utils
from mlstock.utils.utils import time_elapse

logger = logging.getLogger(__name__)


def __get_latest_feature_file():
    import os
    files = os.listdir("data")
    files = [f for f in files if "features" in f]
    files = [os.path.join("data", f) for f in files]
    files.sort(key=lambda f: os.path.getmtime(f))
    files.reverse()
    return files[0]


def main(start_date, end_date, num, is_industry_neutral, option="all", data_file_name=None):
    """
    --
    :param start_date:
    :param end_date:
    :param num:
    :param option: type=str, default="all", help="all|train|data : 所有流程|仅训练|仅加载数据"
    :param data_file_name:
    :return:
    """

    start_time = time.time()

    if option == "train":
        logger.info("仅训练（不做数据清洗），数据预加载自：%s", data_file_name)
        start_time1 = time.time()
        if data_file_name is None:
            logger.info("未提供特征文件名，使用data目录中最新的文件")
            data_file_name = __get_latest_feature_file()
        df_features = pd.read_csv(data_file_name,header=True)
        df_features['trade_date'] = df_features['trade_date'].astype(str)
        factor_names = [item for item in df_features.columns if item not in CODE_DATE] # 只保留特征名
        time_elapse(start_time1, f"从文件中加载训练数据（股票+日期+下期收益+各类特征s）: {data_file_name}")
    else:
        logger.info("加载和清洗数据")

        # 加载特征、基准收益
        df_weekly, factor_names = data_processor.load(start_date, end_date, num)
        assert len(df_weekly) > 0

        # 处理特征，剔除异常等
        df_features = data_processor.process(df_weekly, factor_names, start_date, end_date, is_industry_neutral)
        if option == "data":
            logger.info("仅仅加载和清洗数据，不做训练")
            time_elapse(start_time, "⭐️ 因子加载、因子处理完成")
            return

    logger.info("开始训练数据")

    # 准备训练用数据，需要numpy类型
    X_train = df_features[factor_names].values
    y_train = df_features.target

    # 划分训练集和测试集，测试集占总数据的15%，随机种子为10
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15, random_state=10)

    # train_pct(X_train, y_train)

    train_winloss(X_train, y_train)


def train_pct(X_train, y_train):
    """用岭回归预测下周收益"""

    start_time = time.time()
    best_hyperparam = search_best_hyperparams(X_train, y_train)

    ridge = Ridge(alpha=best_hyperparam)
    ridge.fit(X_train, y_train)
    if not os.path.exists("./model"): os.mkdir("./model")
    model_file_path = f"./model/ridge_{utils.now()}.model"
    joblib.dump(ridge, model_file_path)
    logger.info("训练结果保存到：%s", model_file_path)
    time_elapse(start_time, "⭐️ 岭回归训练完成")


def search_best_hyperparams(X_train, y_train):
    """
    超找最好的超参
    :param X_train:
    :param y_train:
    :return:
    """
    # 做这个是为了人肉看一下最好的岭回归的超参alpha的最优值是啥
    # 是没必要的，因为后面还会用 gridsearch自动跑一下，做这个就是想直观的感受一下
    results = {}
    alpha_scope = np.arange(start=0, stop=1000, step=100)
    for i in alpha_scope:
        # Ridge和Lasso回归分别代表L1和L2的正则化，L1会把系数压缩到0，而L2则不会，同时L1还有挑选特征的作用
        ridge = Ridge(alpha=i)
        results[i] = cross_val_score(ridge, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()

    # 按照value排序：{1: 1, 2: 2, 3: 3} =>[(3, 3), (2, 2), (1, 1)]
    sorted_results = sorted(results.items(), key=lambda x: (x[1], x[0]), reverse=True)

    logger.info("超参数/样本和预测的均方误差：%r", results)
    logger.info("最好的超参数为：%.0f, 对应的最好的均方误差的均值：%.2f",
                sorted_results[0][0],
                sorted_results[0][1])

    # 保存超参的图像
    fig = plt.figure(figsize=(20, 5))
    plt.title('Best Alpha')
    plt.plot(alpha_scope, [results[i] for i in alpha_scope], c="red", label="alpha")
    plt.legend()
    fig.savefig("data/best_alpha.jpg")

    # 自动搜索，暂时保留，上面的代码中，我手工搜索了，还画了图
    # # 用grid search找最好的alpha：[200,205,...,500]
    # # grid search的参数是alpha，岭回归就这样一个参数，用于约束参数的平方和
    # # grid search的入参包括alpha的范围，K-Fold的折数(cv)，还有岭回归评价的函数(负均方误差)
    # grid_search = GridSearchCV(Ridge(),
    #                            {'alpha': alpha_scope},
    #                            cv=5,  # 5折(KFold值)
    #                            scoring='neg_mean_squared_error')
    # grid_search.fit(X_train, y_train)
    # # model = grid_search.estimator.fit(X_train, y_train)
    # logger.info("GridSarch最好的成绩:%.5f", grid_search.best_score_)
    # # 得到的结果是495，确实和上面人肉跑是一样的结果
    # logger.info("GridSarch最好的参数:%.5f", grid_search.best_estimator_.alpha)

    best_hyperparam = sorted_results[0][0]
    return best_hyperparam


def train_winloss(X_train, y_train):
    """
    Xgboost来做输赢判断，参考：https://cloud.tencent.com/developer/article/1656126
    :return:
    """
    start_time = time.time()
    # 创建xgb分类模型实例
    model = XGBClassifier()
    # 待搜索的参数列表空间
    param_lst = {"max_depth": [3, 5, 7, 9],
                 "n_estimators": [*range(10, 110, 20)]}  # [10, 30, 50, 70, 90]
    # 创建网格搜索
    grid_search = GridSearchCV(model,
                               param_grid=param_lst,
                               cv=5,
                               verbose=10,
                               n_jobs=-1)
    # 基于flights数据集执行搜索
    grid_search.fit(X_train, y_train)

    # 输出搜索结果
    logger.debug("GridSearch出最优参数：%r", grid_search.best_estimator_)
    import pdb;
    pdb.set_trace()

    xgboost = XGBClassifier(max_depth=5, min_child_weight=6, n_estimators=300)
    xgboost.fit()
    if not os.path.exists("./model"): os.mkdir("./model")
    model_file_path = f"./model/ridge_{utils.now()}.model"
    joblib.dump(xgboost, model_file_path)
    logger.info("训练结果保存到：%s", model_file_path)

    # 输出搜索结果
    logger.info("", grid_search.best_estimator_)
    time_elapse(start_time, "⭐️ xgboost胜败分类训练完成")


"""
python -m mlstock.ml.train -d
python -m mlstock.ml.train -d -o train
python -m mlstock.ml.train -n 50 -d
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start_date', type=str, default="20090101", help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, default="20220801", help="结束日期")
    parser.add_argument('-n', '--num', type=int, default=100000, help="股票数量，调试用")
    parser.add_argument('-o', '--option', type=str, default="all", help="all|train|data : 所有流程|仅训练|仅加载数据")
    parser.add_argument('-f', '--file', type=str, default=None, help="数据文件")
    parser.add_argument('-in', '--industry_neutral', action='store_true', default=False, help="是否做行业中性处理")
    parser.add_argument('-d', '--debug', action='store_true', default=False, help="是否调试")
    args = parser.parse_args()

    if args.debug:
        print("【调试模式】")
        utils.init_logger(file=True, log_level=logging.DEBUG)
    else:
        utils.init_logger(file=True, log_level=logging.INFO)

    main(start_date=args.start_date,
         end_date=args.end_date,
         num=args.num,
         is_industry_neutral=args.industry_neutral,
         option=args.option)
