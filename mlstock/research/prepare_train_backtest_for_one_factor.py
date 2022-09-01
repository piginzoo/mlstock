import argparse

from mlstock.ml.data import factor_conf, factor_service
from mlstock.research import train_backtest_for_each_factor
from mlstock.utils import utils


def main(factor_name):
    start_date = '20090101'
    split_date = '20190101'
    end_date = '20220901'

    factor_class = factor_conf.get_factor_class_by_name(factor_name)
    _, _, csv_path = factor_service.calculate([factor_class],
                                              start_date,
                                              split_date,
                                              num=50,
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
