import logging

from mlstock.data.stock_info import StocksInfo
from mlstock.factors.factor import FinanceFactor
from mlstock.factors.mixin.fill_mixin import FillMixin
from mlstock.factors.mixin.ttm_mixin import TTMMixin
from mlstock.utils import utils

logger = logging.getLogger(__name__)


class BalanceSheet(FinanceFactor, FillMixin, TTMMixin):
    """
    资产负债表
    """

    # 英文名
    @property
    def name(self):
        return ['total_current_assets',
                'total_none_current_assets',
                'total_assets',
                'total_current_liabilities',
                'total_none_current_liabilities',
                'total_liabilities']

    # tushare 名字，太短，不易理解，需要改成上述
    @property
    def tushare_name(self):
        return ['total_cur_assets',
                'total_nca',
                'total_assets',
                'total_cur_liab',
                'total_ncl',
                'total_liab']

    # 中文名
    @property
    def cname(self):
        return ['流动资产合计', '非流动资产合计', '资产总计', '流动负债合计', '非流动负债合计', '流动负债合计']

    def calculate(self, df_stocks):
        """
        之所有要传入df_stocks，是因为要用他的日期，对每个日期进行TTM填充
        :param df_stocks: 股票周频数据
        :return:
        """
        # 由于财务数据，需要TTM，所以要溯源到1年前，所以要多加载前一年的数据
        start_date_last_year = utils.last_year(self.stocks_info.start_date)

        # 加载资产负债表数据
        df_balancesheet = self.datasource.balance_sheet(self.stocks_info.stocks,
                                                        start_date_last_year,
                                                        self.stocks_info.end_date)

        # 把财务字段改成全名（tushare中的缩写很讨厌）
        df_balancesheet = self._rename_finance_column_names(df_balancesheet)

        # 做财务数据的TTM处理
        df_balancesheet = self.ttm(df_balancesheet, self.name)

        # 按照股票的周频日期，来生成对应的指标（填充周频对应的财务指标）
        df_balancesheet = self.fill(df_stocks, df_balancesheet, self.name)

        # 只保留股票、日期和需要的特征列
        df_balancesheet = self._extract_fields(df_balancesheet)

        return df_balancesheet


# python -m mlstock.factors.balance_sheet
if __name__ == '__main__':
    utils.init_logger(file=False)

    start_date = '20150703'
    end_date = '20190826'
    stocks = ['600000.SH', '002357.SZ', '000404.SZ', '600230.SH']

    from mlstock.data import data_loader
    from mlstock.data.datasource import DataSource

    datasource = DataSource()
    stocks_info = StocksInfo(stocks, start_date, end_date)
    df_stocks = data_loader.weekly(datasource, stocks, start_date, end_date)
    balancesheet = BalanceSheet(datasource, stocks_info)
    df_balancesheet = balancesheet.calculate(df_stocks)
    logger.debug("资产负债表：\n%r", df_balancesheet)
