import argparse

from mlstock.factors.daily_indicator import DailyIndicator
from mlstock.ml.data import factor_conf, factor_service
from mlstock.research import train_backtest_for_each_factor
from mlstock.utils import utils


def main(factor_name):
    start_date = '20090101'
    end_date = '20220901'

    factor_class = factor_conf.get_factor_class_by_name(factor_name)
    # DailyIndicator 始终需要这个因子，因为里面有市值log信息，用于市值中性化
    _, _, csv_path = factor_service.calculate([factor_class,DailyIndicator],
                                              start_date,
                                              end_date,
                                              num=10000000,
                                              is_industry_neutral=True)
    train_backtest_for_each_factor.main(csv_path, factor_name)


"""
python -m mlstock.research.prepare_train_backtest_for_one_factor -f MACD
"""
if __name__ == '__main__':
    utils.init_logger(file=False)
    parser = argparse.ArgumentParser()

    # 数据相关的
    # 数据相关的
    parser.add_argument('-f', '--factor', type=str, default=None, help="因子名字")
    args = parser.parse_args()

    main(args.factor)
