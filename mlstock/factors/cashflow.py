import logging

from mlstock.data.stock_info import StocksInfo
from mlstock.factors import I
from mlstock.factors.factor import FinanceFactor
from mlstock.factors.mixin.fill_mixin import FillMixin
from mlstock.factors.mixin.ttm_mixin import TTMMixin
from mlstock.utils import utils

logger = logging.getLogger(__name__)


class CashFlow(FinanceFactor, FillMixin, TTMMixin):
    """
    现金流量表
    """

    FIELDS_DEF = [
        I('cash_sale_goods_service', 'c_fr_sale_sg', '销售商品、提供劳务收到的现金', '现金流量', ttm=True),
        I('cash_in_subtotal_operate', 'c_inf_fr_operate_a', '经营活动现金流入小计', '现金流量', ttm=True),
        I('cash_paid_goods_service', 'c_paid_goods_s', '购买商品、接受劳务支付的现金', '现金流量', ttm=True),
        I('cash_paid_employees', 'c_paid_to_for_empl', '支付给职工以及为职工支付的现金', '现金流量', ttm=True),
        I('cash_out_subtotal_operate', 'st_cash_out_act', '经营活动现金流出小计', '现金流量', ttm=True),
        I('cash_in_subtotal_invest', 'stot_inflows_inv_act', '投资活动现金流入小计', '现金流量', ttm=True),
        I('cash_out_subtotal_invest', 'stot_out_inv_act', '投资活动现金流出小计', '现金流量', ttm=True),
        I('cash_net_invest', 'n_cashflow_inv_act', '投资活动产生的现金流量净额', '现金流量', ttm=True),
        I('cash_in_subtotal_finance', 'stot_cash_in_fnc_act', '筹资活动现金流入小计', '现金流量', ttm=True),
        I('cash_out_subtotal_finance', 'stot_cashout_fnc_act', '筹资活动现金流出小计', '现金流量', ttm=True),
        I('cash_net_finance', 'n_cash_flows_fnc_act', '筹资活动产生的现金流量净额', '现金流量''', ttm=True)]

    @property
    def data_loader_func(self):
        return self.datasource.cashflow


# python -m mlstock.factors.cashflow
if __name__ == '__main__':
    utils.init_logger(file=False)

    start_date = '20150703'
    end_date = '20190826'
    stocks = ['600000.SH', '002357.SZ', '000404.SZ', '600230.SH']

    CashFlow.test(stocks, start_date, end_date)
