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

    def calculate(self, df_stocks):
        """
        之所有要传入df_stocks，是因为要用他的日期，对每个日期进行TTM填充
        :param df_stocks: 股票周频数据
        :return:
        """

        # 由于财务数据，需要TTM，所以要溯源到1年前，所以要多加载前一年的数据
        start_date_last_year = utils.last_year(self.stocks_info.start_date)

        # 加载收入表数据
        df_income = self.datasource.income(self.stocks_info.stocks,
                                                 start_date_last_year,
                                                 self.stocks_info.end_date)
        # 把财务字段改成全名（tushare中的缩写很讨厌）
        df_income = self._rename_finance_column_names(df_income)

        # 做财务数据的TTM处理
        df_income = self.ttm(df_income, self.name)

        # 按照股票的周频日期，来生成对应的指标（填充周频对应的财务指标）
        df_income = self.fill(df_stocks, df_income, self.name)

        # 只保留股票、日期和需要的特征列
        df_income = self._extract_fields(df_income)

        return df_income


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
