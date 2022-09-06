"""

python -m mlstock.research.backtest_select_top_n \
-s 20190101 -e 20220901 \
-mp model/pct_ridge_20220902112320.model \
-mw model/winloss_xgboost_20220902112813.model \
-d data/factor_20080101_20220901_2954_1299032__industry_neutral_20220902112049.csv
"""
import argparse
import time

from mlstock.ml.backtests.backtest_deliberate import load_datas, run_broker
from mlstock.ml.data import factor_conf
from mlstock.utils import utils


def main(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names):
    """
    先预测出所有的下周收益率、下周涨跌 => df_data，
    然后选出每周的top30 => df_selected_stocks，
    然后使用Broker，来遍历每天的交易，每周进行调仓，并，记录下每周的股票+现价合计价值 => df_portfolio
    最后计算出next_pct_chg、cumulative_pct_chg，并画出plot，计算metrics
    """
    df_data, df_daily, df_index, df_baseline, df_limit, df_calendar = \
        load_datas(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names)

    for top_n in [5, 10, 15, 20, 25, 30, 35, 40]:
        run_broker(df_data, df_daily, df_index, df_baseline, df_limit, df_calendar, start_date, end_date, factor_names,
                   top_n)


if __name__ == '__main__':
    utils.init_logger(file=True)
    parser = argparse.ArgumentParser()

    # 数据相关的
    # parser.add_argument('-t', '--type', type=str, default="simple|backtrader|deliberate")
    parser.add_argument('-s', '--start_date', type=str, default="20190101", help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, default="20220901", help="结束日期")
    parser.add_argument('-d', '--data', type=str, default=None, help="数据文件")
    parser.add_argument('-mp', '--model_pct', type=str, default=None, help="收益率模型")
    parser.add_argument('-mw', '--model_winloss', type=str, default=None, help="涨跌模型")

    args = parser.parse_args()

    factor_names = factor_conf.get_factor_names()

    start_time = time.time()
    main(
        # args.type,
        args.data,
        args.start_date,
        args.end_date,
        args.model_pct,
        args.model_winloss,
        factor_names)
    utils.time_elapse(start_time, "整个回测过程")
