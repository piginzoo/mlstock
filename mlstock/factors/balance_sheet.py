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

    @property
    def data_loader_func(self):
        return self.datasource.balance_sheet


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
