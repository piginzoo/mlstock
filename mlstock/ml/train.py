import argparse
import logging

from mlstock.ml.data import factor_service, factor_conf
from mlstock.ml.train_action.train_pct import TrainPct
from mlstock.ml.train_action.train_winloss import TrainWinLoss
from mlstock.utils import utils

logger = logging.getLogger(__name__)


def main(args):
    utils.check_file_path(args.data)

    df_data = factor_service.load_from_file(args.data)
    df_data = factor_service.extract_features(df_data)

    factor_names = factor_conf.get_factor_names()

    train_pct = TrainPct(factor_names)
    train_winloss = TrainWinLoss(factor_names)

    if args.train == 'all':
        train_pct.evaluate(train_pct.train(df_data))
        train_winloss.evaluate(train_winloss.train(df_data))
        return
    if args.train == 'pct':
        train_pct.evaluate(train_pct.train(df_data))
        return
    if args.train == 'winloss':
        train_winloss.evaluate(train_winloss.train(df_data))
        return


"""
python -m mlstock.ml.train --train all --data data/
"""
if __name__ == '__main__':
    utils.init_logger(file=True, log_level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    # 数据相关的
    parser.add_argument('-s', '--start_date', type=str, default="20090101", help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, default="20220901", help="结束日期")
    parser.add_argument('-n', '--num', type=int, default=100000, help="股票数量，调试用")
    parser.add_argument('-d', '--data', type=str, default=None, help="预先加载的因子数据文件的路径，不再从头计算因子")
    parser.add_argument('-in', '--industry_neutral', action='store_true', default=False, help="是否做行业中性处理")

    # 训练相关的
    parser.add_argument('-t', '--train', type=str, default="all", help="all|pct|winloss : 训练所有|仅训练收益|仅训练涨跌")
    args = parser.parse_args()

    main(args)
