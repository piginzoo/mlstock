from mlstock.data.stock_info import StocksInfo
from mlstock.factors import I
from mlstock.factors.factor import Factor, FinanceFactor

from mlstock.factors.mixin.fill_mixin import FillMixin
from mlstock.factors.mixin.ttm_mixin import TTMMixin
from mlstock.utils import utils
import logging

logger = logging.getLogger(__name__)


class Income(FinanceFactor, FillMixin, TTMMixin):
    """
    利润表
    """
    FIELDS_DEF = [
        I('basic_eps', 'basic_eps', '基本每股收益', '利润表', ttm=True),
        I('diluted_eps', 'diluted_eps', '稀释每股收益', '利润表', ttm=True),
        I('total_revenue', 'total_revenue', '营业总收入', '利润表', ttm=True),
        I('total_cogs', 'total_cogs', '营业总成本', '利润表', ttm=True),
        I('operate_profit', 'operate_profit', '营业利润', '利润表', ttm=True),
        I('none_operate_income', 'non_oper_income', '营业外收入', '利润表', ttm=True),
        I('none_operate_exp', 'non_oper_exp', '营业外支出', '利润表', ttm=True),
        I('total_profit', 'total_profit', '利润总额', '利润表', ttm=True),
        I('net_income', 'n_income', '净利润（含少数股东损益）', '利润表', ttm=True)]

    """
    tushare , is ttm, tushare title, huatai 
    net_after_nr_lp_correct ttm  扣除非经常性损益后的净利润（更正前） | 估值/EPcut/扣除非经常性损益后净利润（TTM）/总市值    
    """

    @property
    def data_loader_func(self):
        return self.datasource.income


# python -m mlstock.factors.income
if __name__ == '__main__':
    utils.init_logger(file=False)

    start_date = '20080101'
    end_date = '20220826'
    stocks = ['600000.SH', '002357.SZ', '000404.SZ', '600230.SH']
    stocks = ['000996.SZ', '002008.SZ', '002014.SZ', '002026.SZ', '002033.SZ',
     '002038.SZ', '002057.SZ', '600728.SH', '000682.SZ', '000697.SZ',
     '000707.SZ', '000718.SZ', '000722.SZ', '000731.SZ', '000737.SZ',
     '000766.SZ', '000778.SZ', '000792.SZ', '000798.SZ', '000812.SZ']
    Income.test(stocks, start_date, end_date)
