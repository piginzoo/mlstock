import argparse
import logging
import time

import joblib

from mlstock.ml.data import factor_service
from mlstock.utils import utils

logger = logging.getLogger(__name__)


def main(args):
    # 查看数据文件和模型文件路径是否正确
    utils.check_file_path(args.data)
    if args.model_pct: utils.check_file_path(args.model_pct)
    if args.model_winloss: utils.check_file_path(args.model_winloss)

    # 加载数据
    df_data = factor_service.load_from_file(args.data)
    original_size = len(df_data)
    original_start_date = df_data.trade_date.min()
    original_end_date = df_data.trade_date.max()
    df_data = df_data[df_data.trade_date >= args.start_date]
    df_data = df_data[df_data.trade_date <= args.end_date]
    logger.debug("数据%s~%s %d行，过滤后=> %s~%s %d行",
                 original_start_date, original_end_date, original_size,
                 args.start_date, args.end_date, len(df_data))

    # 加载模型；如果参数未提供，为None
    model_pct = joblib.load(args.model_pct) if args.model_pct else None
    model_winloss = joblib.load(args.model_winloss) if args.model_winloss else None

    if model_pct:
        start_time = time.time()
        df_data['pred_pct'] = df_data.apply(lambda x: model_pct.predict(x), axis=1)
        utils.time_elapse(start_time, f"预测下期收益: {len(df_data)}行 ")

    if model_pct:
        start_time = time.time()
        df_data['pred_winloss'] = df_data.apply(lambda x: model_winloss.predict(x), axis=1)
        utils.time_elapse(start_time, f"预测下期涨跌: {len(df_data)}行 ")


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

    main(args)
