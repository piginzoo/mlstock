import argparse

from mlstock.ml import train, backtest
from mlstock.ml.data import factor_conf
from mlstock.utils import utils

"""
把因子逐个跑一遍，用其单独训练，单独跑回测，
这样来对比不同因子的效果。
"""
def main(data_path, factor_name):
    start_date = '20090101'
    split_date = '20190101'
    end_date = '20220901'

    factor_names = factor_conf.get_factor_names()

    if factor_name is not None:
        if not factor_name in factor_names:
            raise ValueError(f"因子名[{factor_name}不正确，不在因子列表中]")
        pct_model_path = train.main(data_path, start_date, split_date, 'pct', [factor_name])
        backtest.main(data_path, start_date, split_date, pct_model_path, None, [factor_name])
        backtest.main(data_path, split_date, end_date, pct_model_path, None, [factor_name])
        return

    for factor_name in factor_names:
        pct_model_path = train.main(data_path, start_date, split_date, 'pct', [factor_name])
        backtest.main(data_path, start_date, split_date, pct_model_path, None, [factor_name])
        backtest.main(data_path, split_date, end_date, pct_model_path, None, [factor_name])


# python -m mlstock.research.train_backtest_by_factor -d data/factor_20080101_20220901_2954_1322891_20220829120341.csv
if __name__ == '__main__':
    utils.init_logger(file=False)
    parser = argparse.ArgumentParser()

    # 数据相关的
    parser.add_argument('-d', '--data', type=str, default=None, help="数据文件")
    parser.add_argument('-f', '--factor', type=str, default=None, help="因子名字")
    args = parser.parse_args()

    main(args.data, args.factor)
