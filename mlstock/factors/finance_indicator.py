from mlstock.data.stock_info import StocksInfo
from mlstock.factors import I
from mlstock.factors.factor import FinanceFactor

from mlstock.factors.mixin.fill_mixin import FillMixin
from mlstock.factors.mixin.ttm_mixin import TTMMixin
from mlstock.utils import utils
import logging

logger = logging.getLogger(__name__)


class FinanceIndicator(FinanceFactor, FillMixin, TTMMixin):
    """
    从fina_indicator表中衍生出来指标
    """

    FIELDS_DEF = [
        # 神仔注释掉了，不知为何？我暂且保留
        # ('invturn_days', '存货周转天数')
        # ('arturn_days', '应收账款周转天数')
        # ('inv_turn', '存货周转率')
        I('account_receivable_turnover', 'ar_turn', '应收账款周转率', '营运能力', ttm=True),
        I('current_assets_turnover', 'ca_turn', '流动资产周转率', '营运能力'),
        I('fixed_assets_turnover', 'fa_turn', '固定资产周转率', '营运能力'),
        I('total_assets_turnover', 'assets_turn', '总资产周转率', '营运能力'),
        # 短期偿债能力
        I('current_ratio', 'current_ratio', '流动比率', '短期偿债能力'),
        I('quick_ratio', 'quick_ratio', '速动比率', '短期偿债能力'),
        I('operation_cashflow_to_current_liabilities', 'ocf_to_shortdebt', '经营活动产生的现金流量净额比流动负债', '短期偿债能力'),
        # 长期偿债能力
        I('equity_ratio', 'debt_to_eqt', '产权比率', '长期偿债能力'),
        I('tangible_assets_to_total_liability', 'tangibleasset_to_debt', '有形资产比负债合计', '长期偿债能力'),
        # ('ebit_to_interest','已获利息倍数(EBIT比利息费用)'),
        # 盈利能力
        I('total_profit_to_operate', 'profit_to_op', '利润总额比营业收入', ' 盈利能力'),
        # tushare的中文名为准，tushare的这个英文名有歧义，感觉是年化资产回报率
        # 总资产报酬率是企业一定时期内获得的报酬总额与平均资产总额的比率。 总资产净利率是指公司净利润与平均资产总额的百分比。
        # 而总资产报酬率中的“报酬”可能是任何形式的“利润”，它的范围比较大，有可能是净利润，有可能是息税前利润，也可能是利润总额，是不一定的
        I('annualized_net_return_of_total_assets', 'roa_yearly', '年化总资产净利率', '盈利能力'),
        # 发展能力
        I('total_operate_revenue_yoy', 'tr_yoy', '营业总收入同比增长率(%)', '发展能力'),
        I('operate_revenue_yoy', 'or_yoy', '营业收入同比增长率(%)', '发展能力'),  # 华泰金工：营业收入（最新财报，YTD）同比增长率
        I('total_profit_yoy', 'ebt_yoy', '利润总额同比增长率(%)', '发展能力'),  # tushare叫ebt，我猜是'息税前利润(ebit)的缩写，但是，我还是以中文名为准了',
        I('operate_profit_yoy', 'op_yoy', '营业利润同比增长率(%)', '发展能力'),
        # 这个数据不在默认下载的列中，暂时忽略
        # I('net_profit_yoy', 'q_profit_yoy', '净利润同比增长率', '发展能力'),  # 华泰金工：净利润（最新财报，YTD）同比增长率
        I('operate_cashflow_yoy', 'ocf_yoy', '经营活动产生的现金流量净额同比增长率', '发展能力'),  # 华泰金工：经营性现金流（最新财报，YTD）同比增长率
        # 财务质量
        I('net_profit_deduct_non_recurring_profit_loss', 'profit_dedt', '扣除非经常性损益后的净利润（扣非净利润）', '财务质量', ttm=True),
        # 华泰金工要求的是"扣除非经常性损益后净利润（TTM）/总市值"
        I('operate_cashflow_per_share', 'ocfps', '每股经营活动产生的现金流量净额', '财务质量', ttm=True),
        # 华泰金工要求的是泰金工上的是"经营性现金流（TTM）/总市值"，用这个代替
        I('ROE_TTM', 'roe', '净资产收益率', '财务质量', ttm=True),  # ROE（最新财报，TTM）
        # I('','ROA_yoy', 这个tushare上没有
        I('ROA_TTM', 'roa', '总资产报酬率', '财务质量', ttm=True),  # ROA（最新财报，TTM）
        I('ROE_YOY', 'roe_yoy', '净资产收益率(摊薄)同比增长率', '财务质量'),  # 华泰金工：ROE（最新财报，YTD）同比增长率
    ]

    # 营运能力
    @property
    def data_loader_func(self):
        return self.datasource.fina_indicator


# python -m mlstock.factors.finance_indicator
if __name__ == '__main__':
    utils.init_logger(file=False)

    start_date = '20150703'
    end_date = '20190826'
    stocks = ['600000.SH', '002357.SZ', '000404.SZ', '600230.SH']

    FinanceIndicator.test(stocks, start_date, end_date)
