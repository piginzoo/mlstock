import logging

from mlstock.data.stock_info import StocksInfo
from mlstock.factors.factor import FinanceFactor
from mlstock.factors.mixin.fill_mixin import FillMixin
from mlstock.factors.mixin.ttm_mixin import TTMMixin
from mlstock.utils import utils

logger = logging.getLogger(__name__)


class CashFlow(FinanceFactor, FillMixin, TTMMixin):
    """
    现金流量表
    """

    # 英文名
    @property
    def name(self):
        return ['cash_sale_goods_service',
                'cash_in_subtotal_operate',
                'cash_paid_goods_service',
                'cash_paid_employees',
                'cash_out_subtotal_operate',
                'cash_in_subtotal_invest',
                'cash_out_subtotal_invest',
                'cash_net_invest',
                'cash_in_subtotal_finance',
                'cash_out_subtotal_finance',
                'cash_net_finance']

    # tushare 名字，太短，不易理解，需要改成上述
    @property
    def tushare_name(self):
        return ['c_fr_sale_sg',
                'c_inf_fr_operate_a',
                'c_paid_goods_s',
                'c_paid_to_for_empl',
                'st_cash_out_act',
                'stot_inflows_inv_act',
                'stot_out_inv_act',
                'n_cashflow_inv_act',
                'stot_cash_in_fnc_act',
                'stot_cashout_fnc_act',
                'n_cash_flows_fnc_act']

    # 中文名
    @property
    def cname(self):
        return ['销售商品、提供劳务收到的现金',
                '经营活动现金流入小计',
                '购买商品、接受劳务支付的现金',
                '支付给职工以及为职工支付的现金',
                '经营活动现金流出小计',
                '投资活动现金流入小计',
                '投资活动现金流出小计',
                '投资活动产生的现金流量净额',
                '筹资活动现金流入小计',
                '筹资活动现金流出小计',
                '筹资活动产生的现金流量净额']

    @property
    def data_loader_func(self):
        return self.datasource.cashflow

# python -m mlstock.factors.cashflow
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
    cashflow = CashFlow(datasource, stocks_info)
    df_cashflow = cashflow.calculate(df_stocks)
    logger.debug("收入表：\n%r", df_cashflow)