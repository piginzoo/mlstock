from mlstock.data.stock_info import StocksInfo
from mlstock.factors.factor import Factor, FinanceFactor

from mlstock.factors.mixin.fill_mixin import FillMixin
from mlstock.factors.mixin.ttm_mixin import TTMMixin
from mlstock.utils import utils
import logging

logger = logging.getLogger(__name__)

PERIOD = 12


class Income(FinanceFactor, FillMixin, TTMMixin):
    """
    利润表
    """

    # 英文名
    @property
    def name(self):
        return ['basic_eps',
                'diluted_eps',
                'total_revenue',
                'total_cogs',
                'operate_profit',
                'none_operate_income',
                'none_operate_exp',
                'total_profit',
                'net_income']

    # tushare 名字，太短，不易理解，需要改成上述
    @property
    def tushare_name(self):
        return ['basic_eps',
                'diluted_eps',
                'total_revenue',
                'total_cogs',
                'operate_profit',
                'non_oper_income',
                'non_oper_exp',
                'total_profit',
                'n_income']
    """
    tushare , is ttm, tushare title, huatai 
    net_after_nr_lp_correct ttm  扣除非经常性损益后的净利润（更正前） | 估值/EPcut/扣除非经常性损益后净利润（TTM）/总市值
    
    """

    def calculate(self, stock_data):
        pass



    # 中文名
    @property
    def cname(self):
        return ['基本每股收益',
                '稀释每股收益',
                '营业总收入',
                '营业总成本',
                '营业利润',
                '营业外收入',
                '营业外支出',
                '利润总额',
                '净利润（含少数股东损益）']

    @property
    def data_loader_func(self):
        return self.datasource.income


# python -m mlstock.factors.income
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
    income = Income(datasource, stocks_info)
    df_income = income.calculate(df_stocks)
    logger.debug("收入表：\n%r", df_income)
