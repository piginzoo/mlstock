import argparse
import logging
import time

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pandas import DataFrame

from mlstock.const import TOP_30
from mlstock.ml import load_and_filter_data
from mlstock.ml.data import factor_conf
from mlstock.utils import utils

logger = logging.getLogger(__name__)


def main(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names):
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
    # 从csv因子数据文件中加载数据
    df_data = load_and_filter_data(data_path, start_date, end_date)

    # 加载模型；如果参数未提供，为None
    # 查看数据文件和模型文件路径是否正确
    if model_pct_path: utils.check_file_path(model_pct_path)
    if model_winloss_path: utils.check_file_path(model_winloss_path)
    model_pct = joblib.load(model_pct_path) if model_pct_path else None
    model_winloss = joblib.load(model_winloss_path) if model_winloss_path else None

    if model_pct:
        start_time = time.time()
        X = df_data[factor_names]
        df_data['pct_pred'] = model_pct.predict(X)
        utils.time_elapse(start_time, f"预测下期收益: {len(df_data)}行 ")

    if model_winloss:
        start_time = time.time()
        X = df_data[factor_names]
        df_data['winloss_pred'] = model_winloss.predict(X)
        utils.time_elapse(start_time, f"预测下期涨跌: {len(df_data)}行 ")

    # 按照预测的结果，来选择股票
    df_pct = select_stocks_by_pred_and_calcuate_portfolio(df_data)

    # 画出回测图
    plot(df_pct, start_date, end_date, factor_names)


def select_stocks_by_pred_and_calcuate_portfolio(df):
    """
    根据预测收益率，选择股票，并且计算top30的组合收益率
    :param df:
    :return:
    """

    # 先把所有预测为跌的全部过滤掉
    original_size = len(df)
    df = df[df.winloss_pred == 1]
    logger.debug("根据涨跌模型结果，过滤数据 %d=>%d", original_size, len(df))

    # 先按照日期 + 下周预测收益，按照降序排
    df = df.sort_values(['trade_date', 'pct_pred'], ascending=False)

    # 按照日期分组，每组里面取前30，然后算收益率，作为组合资产的收益率
    # 注意！这里是下期收益"next_pct_chg"的均值，实际上是提前了一期（这个细节可以留意一下）
    # df_pct = df.groupby('trade_date')['next_pct_chg', 'next_pct_chg_baseline'].apply(
    #     lambda df_group: df_group[0:30].mean())
    df_groups = df.groupby('trade_date')

    # 用于保存预测收益率
    df_portfolio_pct = DataFrame()

    # 用于保存选择后的股票，特别是他们的下期的实际收益率
    df_selected_stocks = DataFrame()

    for date, df_group in df_groups:

        # 根据 "预测收益率" 选出收益率top30
        df_top30 = df_group.iloc[0:TOP_30, :]

        # 根据 "实际收益率" 对这些选中股票求平均收益率（作为资产组合的收益率）
        next_pct_chg_mean = np.mean(df_top30.next_pct_chg.values)

        # 对基准的实际收益率也求一个平均（其实她们每个股票的这个值都是一样的，相加再去平均数，其实还是原来的数）
        next_pct_chg_baseline_mean = np.mean(df_top30.next_pct_chg_baseline.values)

        df_portfolio_pct = df_portfolio_pct.append([[date, next_pct_chg_mean, next_pct_chg_baseline_mean]])
        df_top30['trade_date'] = date
        df_selected_stocks = df_selected_stocks.append(df_top30)

    # 处理选中的股票的信息，保存下来，其实没啥用，就是存一下，方便细排查
    df_selected_stocks = df_selected_stocks[
        ['trade_date', 'ts_code', 'target', 'pct_pred', 'next_pct_chg', 'next_pct_chg_baseline']]
    df_selected_stocks.columns = [
        'trade_date', 'ts_code', 'target', 'pct_pred', 'next_pct_chg', 'next_pct_chg_baseline']
    df_selected_stocks.to_csv("data/top30.csv", header=0)

    # 组合的收益率情况
    df_portfolio_pct.columns = ['trade_date', 'next_pct_chg', 'next_pct_chg_baseline']
    df_portfolio_pct[['cumulative_pct_chg', 'cumulative_pct_chg_baseline']] = \
        df_portfolio_pct[['next_pct_chg', 'next_pct_chg_baseline']].apply(lambda x: (x + 1).cumprod() - 1)

    return df_portfolio_pct.reset_index()


def plot(df, start_date, end_date, factor_names):
    """
    1. 每期实际收益
    2. 每期实际累计收益
    3. 基准累计收益率
    4. 上证指数
    :param df:
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    x = df.trade_date.values
    y1 = df.next_pct_chg.values
    y2 = df.cumulative_pct_chg.values
    y3 = df.cumulative_pct_chg_baseline.values
    color_y1 = '#2A9CAD'
    color_y2 = "#FAB03D"
    color_y3 = "#FAFF0D"
    title = '资产组合收益率及累积收益率'
    label_x = '周'
    label_y1 = '资产组合周收益率'
    label_y2 = '资产组合累积收益率'
    label_y3 = '基准累积收益率'
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    plt.xticks(rotation=60)
    ax2 = ax1.twinx()  # 做镜像处理

    ax1.bar(x=x, height=y1, label=label_y1, color=color_y1, alpha=0.7)
    ax2.plot(x, y2, color=color_y2, ms=10, label=label_y2)
    ax2.plot(x, y3, color=color_y3, ms=10, label=label_y3)

    ax1.set_xlabel(label_x)  # 设置x轴标题
    ax1.set_ylabel(label_y1)  # 设置Y1轴标题
    ax2.set_ylabel(label_y2 + "/" + label_y3)  # 设置Y2轴标题
    ax1.grid(False)
    ax2.grid(False)

    # 12周间隔，3个月相当于，为了让X轴稀疏一些，太密了，如果不做的话
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(12))

    # 添加标签
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(title)  # 添加标题
    plt.grid(axis="y")  # 背景网格

    # 保存图片
    factor = '' if len(factor_names) > 1 else factor_names[0]
    save_path = 'data/plot_{}_{}_{}.jpg'.format(factor, start_date, end_date)
    plt.savefig(save_path)
    plt.show()


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

    factor_names = factor_conf.get_factor_names()

    main(args.data,
         args.start_date,
         args.end_date,
         args.model_pct,
         args.model_winloss,
         factor_names)
