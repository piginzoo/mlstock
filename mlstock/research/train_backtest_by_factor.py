import argparse

from mlstock.ml import load_and_filter_data, train, backtest
from mlstock.ml.data import factor_conf
from mlstock.utils import utils


def main(data_path):
    start_date = '20090101'
    split_date = '20190101'
    end_date = '20220901'

    factor_names = factor_conf.get_factor_names()
    for factor_name in factor_names:
        pct_model_path = train.main(data_path, start_date, split_date, 'pct', [factor_name])
        backtest.main(data_path, split_date, end_date, pct_model_path, None, [factor_name])

# python -m mlstock.research.train_backtest_by_factor factor_20080101_20220901_2954_1322891_20220829120341.csv
if __name__ == '__main__':
    utils.init_logger(file=False)
    parser = argparse.ArgumentParser()

    # 数据相关的
    parser.add_argument('-d', '--data', type=str, default=None, help="数据文件")
    args = parser.parse_args()

    main(args.data)
