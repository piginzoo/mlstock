import argparse
import logging

from mlstock.ml import load_and_filter_data
from mlstock.ml.data import factor_conf
from mlstock.ml.train_action.train_pct import TrainPct
from mlstock.ml.train_action.train_winloss import TrainWinLoss
from mlstock.utils import utils

logger = logging.getLogger(__name__)


def main(data_path, start_date, end_date, train_type, factor_names):
    """
    训练
    :param data_path: 数据（因子）csv文件的路径
    :param start_date: 训练数据的开始日期
    :param end_date: 训练数据的结束日子
    :param train_type: all|pct|winloss，方便单独训练
    :param factor_names: 因子的名称，用于过滤出X
    :return:
    """
    # 从csv文件中加载数据，现在统一成从文件加载了，之前还是先清洗，用得到的dataframe，但过程很慢，
    # 改成先存成文件，再从文件中加载，把过程分解了，方便做pipeline
    df_data = load_and_filter_data(data_path, start_date, end_date)

    # 收益率回归模型
    train_pct = TrainPct(factor_names)

    # 涨跌分类模型
    train_winloss = TrainWinLoss(factor_names)

    # 回归+分类
    if train_type == 'all':
        return [train_pct.train(df_data), train_winloss.train(df_data)]

    # 仅回归
    if train_type == 'pct':
        return train_pct.train(df_data)

    # 仅分类
    if train_type == 'winloss':
        return train_winloss.train(df_data)

    raise ValueError(f"无法识别训练类型:{train_type}")


"""
python -m mlstock.ml.train --train all --data data/
"""
if __name__ == '__main__':
    utils.init_logger(file=True, log_level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    # 数据相关的
    parser.add_argument('-s', '--start_date', type=str, default="20090101", help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, default="20190101", help="结束日期")
    parser.add_argument('-n', '--num', type=int, default=100000, help="股票数量，调试用")
    parser.add_argument('-d', '--data', type=str, help="预先加载的因子数据文件的路径，不再从头计算因子")
    parser.add_argument('-in', '--industry_neutral', action='store_true', default=False, help="是否做行业中性处理")

    # 训练相关的
    parser.add_argument('-t', '--train', type=str, default="all", help="all|pct|winloss : 训练所有|仅训练收益|仅训练涨跌")
    args = parser.parse_args()
    factor_names = factor_conf.get_factor_names()
    logger.info("训练使用的特征 %d 个：%r", len(factor_names), factor_names)

    main(args.data, args.start_date, args.end_date, args.train, factor_names)
