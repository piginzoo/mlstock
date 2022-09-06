import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, \
    recall_score, f1_score

from mlstock.ml import load_and_filter_data
from mlstock.ml.data import factor_service, factor_conf
from mlstock.ml.data.factor_service import extract_features
from mlstock.utils import utils

logger = logging.getLogger(__name__)


def classification_metrics(df, model):
    """
    https://ningshixian.github.io/2020/08/24/sklearn%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87/
    """

    X = extract_features(df)
    y = df.target.apply(lambda x: 1 if x > 0 else 0)

    y_pred = model.predict(X)
    metrics = {}
    metrics['accuracy'] = accuracy_score(y, y_pred)
    metrics['precision'] = precision_score(y, y_pred)
    metrics['recall'] = recall_score(y, y_pred)
    metrics['f1'] = f1_score(y, y_pred)

    logger.debug("计算出回归指标：%r", metrics)
    return metrics


def regression_metrics(df, model):
    """
    https://blog.csdn.net/u012735708/article/details/84337262
    """
    metrics = {}

    df['y'] = df['target']
    X = extract_features(df)

    df['y_pred'] = model.predict(X)

    metrics['corr'] = df[['y', 'y_pred']].corr().iloc[0, 1]  # 测试标签y和预测y_pred相关性，到底准不准啊

    # 看一下rank的IC，不看值相关性，而是看排名的相关性
    df['y_rank'] = df.y.rank(ascending=False)  # 并列的默认使用排名均值
    df['y_pred_rank'] = df.y_pred.rank(ascending=False)
    metrics['rank_corr'] = df[['y_rank', 'y_pred_rank']].corr().iloc[0, 1]

    metrics['RMSE'] = np.sqrt(mean_squared_error(df.y, df.y_pred))

    metrics['MA'] = mean_absolute_error(df.y, df.y_pred)

    metrics['R2'] = r2_score(df.y, df.y_pred)

    logger.debug("计算出分类指标：%r", metrics)
    return metrics


def factor_weights(model):
    """
    显示权重的影响
    :param model:
    :return:
    """

    param_weights = model.coef_
    param_names = factor_conf.get_factor_names()
    df = pd.DataFrame({'names': param_names, 'weights': param_weights})
    df = df.reindex(df.weights.abs().sort_values().index)
    logger.info("参数和权重排序：")
    logger.info(df.to_string().replace('\n', '\n\t'))


def main(args):
    # 查看数据文件和模型文件路径是否正确
    if args.model_pct: utils.check_file_path(args.model_pct)
    if args.model_winloss: utils.check_file_path(args.model_winloss)

    df_data = load_and_filter_data(args.data, args.start_date, args.end_date)

    logger.info("周收益平均值：%.2f%%", df_data.target.mean() * 100)
    logger.info("周收益标准差：%.2f%%", df_data.target.std() * 100)
    logger.info("周收益中位数：%.2f%%", df_data.target.median() * 100)
    logger.info("绝对值平均值：%.2f%%", df_data.target.abs().mean() * 100)
    logger.info("绝对值标准差：%.2f%%", df_data.target.abs().std() * 100)
    logger.info("绝对值中位数：%.2f%%", df_data.target.abs().median() * 100)

    # 加载模型；如果参数未提供，为None
    model_pct = joblib.load(args.model_pct) if args.model_pct else None
    model_winloss = joblib.load(args.model_winloss) if args.model_winloss else None

    if model_pct:
        factor_weights(model_pct)
        regression_metrics(df_data, model_pct)

    if model_winloss:
        classification_metrics(df_data, model_winloss)


"""
python -m mlstock.ml.evaluate \
-s 20190101 -e 20220901 \
-mp model/pct_ridge_20220902112320.model \
-mw model/winloss_xgboost_20220902112813.model \
-d data/factor_20080101_20220901_2954_1299032__industry_neutral_20220902112049.csv
"""
if __name__ == '__main__':
    utils.init_logger(file=True)
    parser = argparse.ArgumentParser()

    # 数据相关的
    parser.add_argument('-s', '--start_date', type=str, default="20190101", help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, default="20220901", help="结束日期")
    parser.add_argument('-d', '--data', type=str, default=None, help="数据文件")
    parser.add_argument('-mp', '--model_pct', type=str, default=None, help="收益率模型")
    parser.add_argument('-mw', '--model_winloss', type=str, default=None, help="涨跌模型")

    args = parser.parse_args()

    main(args)
