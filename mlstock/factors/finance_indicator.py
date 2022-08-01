from mlstock.data.stock_info import StocksInfo
from mlstock.factors.factor import Factor, FinanceFactor

from mlstock.factors.mixin.fill_mixin import FillMixin
from mlstock.factors.mixin.ttm_mixin import TTMMixin
from mlstock.utils import utils
import logging

logger = logging.getLogger(__name__)

PERIOD = 12


class FinanceIndicator(FinanceFactor, FillMixin, TTMMixin):
    """
    从fina_indicator表中衍生出来指标
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
        return ['profit_dedt',
                'ocfps',
                'or_yoy',
                'q_profit_yoy'
                'ocf_yoy',
                'roe_yoy']

    # 中文名
    @property
    def cname(self):
        return ['扣除非经常性损益后的净利润（扣非净利润）',  # 华泰金工要求的是"扣除非经常性损益后净利润（TTM）/总市值"
                '每股经营活动产生的现金流量净额',  # 华泰金工上的是"经营性现金流（TTM）/总市值"，用这个代替
                '营业收入同比增长率',  # 华泰金工：营业收入（最新财报，YTD）同比增长率
                '净利润同比增长率',  # 华泰金工：净利润（最新财报，YTD）同比增长率
                '经营活动产生的现金流量净额同比增长率',  # 华泰金工：经营性现金流（最新财报，YTD）同比增长率
                '净资产收益率(摊薄)同比增长率'  # 华泰金工：ROE（最新财报，YTD）同比增长率
                'ROE',
                'ROE_TTM',
                'ROA',
                'ROA_TTM'
                ]

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
