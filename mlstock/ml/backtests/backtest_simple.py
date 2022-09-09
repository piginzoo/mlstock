import argparse
import logging

from mlstock.const import TOP_30
from mlstock.data.datasource import DataSource
from mlstock.ml.backtests import plot, predict, select_top_n
from mlstock.ml.backtests.metrics import metrics
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

    datasource = DataSource()

    df_data = predict(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names)
    df_limit = datasource.limit_list()

    df_selected_stocks = select_top_n(df_data, df_limit,TOP_30)
    df_selected_stocks = df_selected_stocks.reset_index(drop=True)


    # 组合的收益率情况
    df_portfolio = df_selected_stocks.groupby('trade_date')[['next_pct_chg', 'next_pct_chg_baseline']].mean().reset_index()

    # 拼接上指数
    df_index = datasource.index_weekly('000001.SH', start_date, end_date)
    df_index = df_index[['trade_date', 'close']]
    df_index = df_index.rename(columns={'close': 'index_close'})
    df_portfolio = df_portfolio.merge(df_index, how='left', on='trade_date')

    df_portfolio.columns = ['trade_date', 'next_pct_chg', 'next_pct_chg_baseline','index_close']
    df_portfolio[['cumulative_pct_chg', 'cumulative_pct_chg_baseline']] = \
        df_portfolio[['next_pct_chg', 'next_pct_chg_baseline']].apply(lambda x: (x + 1).cumprod() - 1)

    df_portfolio.sort_values('trade_date')

    # 画出回测图
    plot(df_portfolio, f"data/plot_simple_{start_date}_{end_date}")

    # 计算各项指标
    metrics(df_portfolio)


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
