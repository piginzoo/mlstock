import logging

from mlstock.data.stock_info import StocksInfo
from mlstock.factors import I
from mlstock.factors.factor import FinanceFactor
from mlstock.factors.mixin.fill_mixin import FillMixin
from mlstock.factors.mixin.ttm_mixin import TTMMixin
from mlstock.utils import utils

logger = logging.getLogger(__name__)


class BalanceSheet(FinanceFactor, FillMixin, TTMMixin):
    """
    资产负债表
    """

    FIELDS_DEF = [
        I('total_current_assets', 'total_cur_assets', '流动资产合计', '资产负债'),
        I('total_none_current_assets', 'total_nca', '非流动资产合计', '资产负债'),
        I('total_assets', 'total_assets', '资产总计', '资产负债'),
        I('total_current_liabilities', 'total_cur_liab', '流动负债合计', '资产负债'),
        I('total_none_current_liabilities', 'total_ncl', '非流动负债合计', '资产负债'),
        I('total_liabilities', 'total_liab', '流动负债合计', '资产负债')]

    @property
    def data_loader_func(self):
        return self.datasource.balance_sheet

# python -m mlstock.factors.balance_sheet
if __name__ == '__main__':
    utils.init_logger(file=False)

    start_date = '20150703'
    end_date = '20190826'
    stocks = ['600000.SH', '002357.SZ', '000404.SZ', '600230.SH']

    BalanceSheet.test(stocks, start_date, end_date)
