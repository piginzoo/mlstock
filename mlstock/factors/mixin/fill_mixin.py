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
        # 按照公布日期倒序排（日期从新到旧）
        df_finance.sort_values('ann_date', ascending=False)

        # 只用股票的ts_code和trad_code，两列，去取得对应的财务指标
        return df_stocks.groupby(by=['ts_code', 'trade_date']).apply(self.handle_one_stock,
                                                                     df_finance,
                                                                     finance_column_names)

    # @logging_time
    def handle_one_stock(self, df_stock, df_finance, finance_column_names):
        """
        处理一只股票得财务数据填充
        :param df: 一只股票的数据的周频的一期数据
        """
        # 只是用股票代码和日期信息
        df_stock = df_stock[['ts_code', 'trade_date']]
        df_stock[finance_column_names] = self.__find_nearest_values(df_stock, df_finance, finance_column_names)
        return df_stock

    def __find_nearest_values(self, df_stock, df_finance, finance_column_names):
        """
        找到离我最旧最近的一天的财务数据作为我当天的财务数据，
        举例：当前是8.15，找到离我最近的是7.31号发布的6.30号的半年数据，那么就用这个数据，作为我(8.15)的财务数据
        :param df_stock:
        :param df_finance:
        :param finance_column_names:
        :return:
        """
        # 这只股票的代码
        ts_code = df_stock.iloc[0]['ts_code']
        # 这只股票当前的日期
        trade_date = df_stock.iloc[0]['trade_date']
        # 过滤出这只股票的财务数据
        df_finance = df_finance[df_finance['ts_code'] == ts_code]

        # 按照日子挨个查找，找到离我最旧最近的公布日，返回其公布日对应的财务数据
        for _, row in df_finance.iterrows():
            if trade_date >= row['ann_date']:
                # logger.debug("找到股票[%s]的周频日期[%s]的财务日为[%s]的填充数据", ts_code, trade_date, row['ann_date'])
                return row[finance_column_names]
        return None


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
    logger.info("原始数据：\n%r\n%r", df_stock, df_finance)
    df = FillMixin().fill(df_stock, df_finance, finance_column_names)
    logger.info("填充完数据：\n%r", df)
