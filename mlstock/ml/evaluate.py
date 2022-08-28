import argparse
import logging

import joblib
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, \
    recall_score, f1_score

from mlstock.ml.data import factor_service, factor_conf
from mlstock.utils import utils

logger = logging.getLogger(__name__)


def _extract_features(df):
    return df[factor_conf.get_factor_names()]


def classification_metrics(df, model):
    """
    https://ningshixian.github.io/2020/08/24/sklearn%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87/
    """

    X = _extract_features(df)
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
    X = _extract_features(df)

    df['y_pred'] = model.predict(X)

    metrics['corr'] = df[['next_pct_chg', 'y_pred']].corr().iloc[0, 1]  # 测试标签y和预测y_pred相关性，到底准不准啊

    # 看一下rank的IC，不看值相关性，而是看排名的相关性
    df['y_rank'] = df.next_pct_chg.rank(ascending=False)  # 并列的默认使用排名均值
    df['y_pred_rank'] = df.y_pred.rank(ascending=False)
    metrics['rank_corr'] = df[['y_rank', 'y_pred_rank']].corr().iloc[0, 1]

    metrics['RMSE'] = np.sqrt(mean_squared_error(df.y, df.y_pred))

    metrics['MA'] = mean_absolute_error(df.y, df.y_pred)

    metrics['R2'] = r2_score(df.y, df.y_pred)

    logger.debug("计算出分类指标：%r", metrics)
    return metrics


def main(args):
    # 查看数据文件和模型文件路径是否正确
    utils.check_file_path(args.data)
    if args.model_pct: utils.check_file_path(args.model_pct)
    if args.model_winloss: utils.check_file_path(args.model_winloss)

    # 加载数据
    df_data = factor_service.load_from_file(args.data)
    original_size = len(df_data)
    original_start_date = df_data.trade_date.min()
    original_end_date = df_data.trade_date.max()
    df_data = df_data[df_data.trade_date >= args.start_date]
    df_data = df_data[df_data.trade_date <= args.end_date]
    logger.debug("数据%s~%s %d行，过滤后=> %s~%s %d行",
                 original_start_date, original_end_date, original_size,
                 args.start_date, args.end_date, len(df_data))

    # 加载模型；如果参数未提供，为None
    model_pct = joblib.load(args.model_pct) if args.model_pct else None
    model_winloss = joblib.load(args.model_winloss) if args.model_winloss else None

    if model_pct:
        regression_metrics(df_data, model_pct)

    if model_winloss:
        classification_metrics(df_data, model_winloss)


"""
python -m mlstock.ml.evaluate \
-s 20190101 -e 20220901 \
-mp model/pct_ridge_20220828190251.model \
-mw model/winloss_xgboost_20220828190259.model \
-d data/processed_industry_neutral_20080101_20220901_20220828152110.csv
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
