import logging
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from pandas import DataFrame

from mlstock.const import TOP_30
from mlstock.ml import load_and_filter_data
from mlstock.utils import utils

logger = logging.getLogger(__name__)


def select_top_n(df, df_limit):
    """
    :param df:
    :param df_limit:
    :return:
    """

    # 先把所有预测为跌的全部过滤掉
    original_size = len(df)
    df = df[df.winloss_pred == 1]
    logger.debug("根据涨跌模型结果，过滤数据 %d=>%d", original_size, len(df))

    # 剔除那些涨停的股票,参考：https://stackoverflow.com/questions/32652718/pandas-find-rows-which-dont-exist-in-another-dataframe-by-multiple-columns
    df_limit = df_limit[['trade_date', 'ts_code', 'limit']]
    original_size = len(df)
    df = df.merge(df_limit, on=['ts_code', 'trade_date'], how='left', indicator=True)
    df = df[df._merge == 'left_only']
    logger.debug("根据涨跌停信息，过滤数据 %d=>%d", original_size, len(df))

    # 先按照日期 + 下周预测收益，按照降序排
    df = df.sort_values(['trade_date', 'pct_pred'], ascending=False)

    # 按照日期分组，每组里面取前30，然后算收益率，作为组合资产的收益率
    # 注意！这里是下期收益"next_pct_chg"的均值，实际上是提前了一期（这个细节可以留意一下）
    # .reset_index(level=0, drop=True)
    df_selected_stocks = df.groupby('trade_date').apply(lambda grp: grp.nlargest(TOP_30, 'pct_pred'))
    logger.debug("按照预测收益率挑选出%d条股票信息", len(df_selected_stocks))

    return df_selected_stocks.reset_index(drop=True)


def predict(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names):
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
    return df_data


# 画A股,参考：https://deepinout.com/matplotlib/matplotlib-axis/matplotlib-the-hidden-scale-mode-shows-the-three-y-axes.html
# 创建第三条Y轴，把第三条Y轴的其它三边隐藏起来，只留下右边显示
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


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
    y4 = df.index_close.values

    color_y1 = '#2A9CAD'
    color_y2 = "#FAB03D"
    color_y3 = "#D3D3D3"
    color_y4 = "#008000"

    title = '资产组合收益率及累积收益率'

    label_x = '周'
    label_y1 = '资产组合周收益率'
    label_y2 = '资产组合累积收益率'
    label_y3 = '基准累积收益率'
    label_y4 = '上证指数'

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    plt.xticks(rotation=60)
    ax2 = ax1.twinx()  # 做镜像处理

    ax1.bar(x=x, height=y1, label=label_y1, color=color_y1, alpha=0.7)
    ax1.set_xlabel(label_x)  # 设置x轴标题
    ax1.set_ylabel(label_y1)  # 设置Y1轴标题
    ax1.grid(False)
    # 12周间隔，3个月相当于，为了让X轴稀疏一些，太密了，如果不做的话
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(12))

    ax2.plot(x, y2, color=color_y2, ms=10, label=label_y2)
    ax2.plot(x, y3, color=color_y3, ms=10, label=label_y3)
    ax2.set_ylabel(label_y2 + "/" + label_y3)  # 设置Y2轴标题
    ax2.grid(False)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(12))

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)
    p3, = ax3.plot(x, y4, color=color_y4, ms=10, label=label_y4)
    ax3.set_ylabel(label_y4)
    ax3.grid(False)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax3.yaxis.label.set_color(p3.get_color())

    # 添加标签
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax3.legend(loc='lower right')

    plt.title(title)  # 添加标题
    plt.grid(axis="y")  # 背景网格

    # 保存图片
    factor = '' if len(factor_names) > 1 else factor_names[0]
    save_path = 'data/plot_{}_{}_{}.jpg'.format(factor, start_date, end_date)
    plt.savefig(save_path)
    plt.show()
