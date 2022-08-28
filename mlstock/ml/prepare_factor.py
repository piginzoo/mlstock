import argparse

from mlstock.ml.data import factor_service
from mlstock.utils import utils


def load_data(args):
    start_date = args.start_date
    end_date = args.end_date
    num = args.num
    is_industry_neutral = args.industry_neutral

    # 那么就需要从新计算了
    df_weekly, factor_names = factor_service.calculate(start_date, end_date, num, is_industry_neutral)
    return df_weekly, factor_names
"""
python -m mlstock.ml.prepare_factor -n 50 -in -s 20080101 -e 20220901
"""
if __name__ == '__main__':
    utils.init_logger(file=True)

    parser = argparse.ArgumentParser()

    # 数据相关的
    parser.add_argument('-s', '--start_date', type=str, default="20090101", help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, default="20220901", help="结束日期")
    parser.add_argument('-n', '--num', type=int, default=100000, help="股票数量，调试用")

    parser.add_argument('-in', '--industry_neutral', action='store_true', default=False, help="是否做行业中性处理")

    args = parser.parse_args()

    df_data, factor_names = load_data(args)
