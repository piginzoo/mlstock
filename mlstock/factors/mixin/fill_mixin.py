from mlstock.utils import utils

import logging

from mlstock.utils.utils import logging_time

logger = logging.getLogger(__name__)


class FillMixin:

    @logging_time("间隔填充")
    def fill(self, df_stocks, df_finance, finance_column_names):
        """
        将离散的单个的财务TTM信息，反向填充到周频的交易数据里去，
        比如周频，日期为7.22（周五），那对应的财务数据是6.30日的，
        所以要用6.30的TTM数据去填充到这个7.22日上去。

        这个是配合TTM使用，TTM对公告日期进行了TTM，但只有公告日当天的，
        但是，我们需要的是周频的那天的TTM，考虑到财报的滞后性，
        某天的TTM值，'合理'的是他之前的最后一个公告日对应的TTM。
        所以，才需要这个fill函数，来填充每一个周五的TTM。

        :param df_stocks: df是原始的周频数据，以周五的日期为准
        :param df_finance: 财务数据，只包含财务公告日期，要求之前已经做了ttm数据
        """
        # 按照公布日期倒序排（日期从新到旧）,<=== '倒'序很重要，这样才好找到对应日的财务数据
        df_finance = df_finance.sort_values('ann_date')
        # 开始做join合并，注意注意，用outer，外连接，这样就不会落任何两边的日期（财务的，和，股票交易数据的）
        df_merge = df_stocks.merge(df_finance,
                                   how="outer",
                                   left_on=['ts_code','trade_date'],
                                   right_on=['ts_code','ann_date'])
        # import pdb;pdb.set_trace()
        # 因为做了join合并，所以必须要做一个日期排序，因为日期是乱的
        df_merge = df_merge.sort_values(['ts_code','trade_date'])
        # 然后，就可以对指定的财务字段，进行填充了，先用ffill，后用bfill
        df_merge[finance_column_names] = df_merge.groupby('ts_code').ffill().bfill()[finance_column_names]
        # 取close这一列（它在股票数据里肯定不是空，但被outer-join后，财务日期对应的close肯定是空，用它来过滤），来过滤掉无效数据
        df_merge = df_merge[~df_merge['close'].isna()]
        # 只取需要的列
        df = df_merge[['ts_code','trade_date']+finance_column_names]
        return df


# python -m mlstock.factors.mixin.fill_mixin
if __name__ == '__main__':
    utils.init_logger(file=False)

    start_date = '20150703'
    end_date = '20190826'
    stocks = ['600000.SH', '002357.SZ', '000404.SZ', '600230.SH']
    finance_column_names = ['basic_eps', 'diluted_eps']

    import tushare as ts
    import pandas as pd

    pro = ts.pro_api()

    df_finance_list = []
    df_stock_list = []
    for ts_code in stocks:
        df_finance = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date,
                                fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,basic_eps,diluted_eps')
        df_finance_list.append(df_finance)
        df_stock = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df_stock_list.append(df_stock)
    df_finance = pd.concat(df_finance_list)
    df_stock = pd.concat(df_stock_list)
    logger.info("原始数据：\n股票数据\n%r\n财务数据：%r", df_stock, df_finance)
    df = FillMixin().fill(df_stock, df_finance, finance_column_names)
    logger.info("填充完数据：\n%r", df)
