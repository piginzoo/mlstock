from mlstock.factors.factor import Factor
import numpy as np
import pandas as pd

PERIOD = 12


class Income(Factor):
    """
    利润表
    """

    # 英文名
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
        df_balancesheet = self.datasource.income(self.stocks_info.stocks,
                                                        self.stocks_info.start_date,
                                                        self.stocks_info.end_date)
        return self._extract_fields(df_balancesheet)
