from mlstock.ml.data import factor_service

import logging

from mlstock.utils import utils

logger = logging.getLogger(__name__)


def load_and_filter_data(data_path, start_date, end_date):
    # 加载数据
    utils.check_file_path(data_path)

    df_data = factor_service.load_from_file(data_path)
    original_size = len(df_data)
    original_start_date = df_data.trade_date.min()
    original_end_date = df_data.trade_date.max()
    df_data = df_data[df_data.trade_date >= start_date]
    df_data = df_data[df_data.trade_date <= end_date]
    logger.debug("数据%s~%s %d行，过滤后=> %s~%s %d行",
                 original_start_date, original_end_date, original_size,
                 start_date, end_date, len(df_data))
    return df_data
