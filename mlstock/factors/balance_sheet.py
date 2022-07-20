from mlstock.factors.factor import Factor
import numpy as np
import pandas as pd

PERIOD = 12


class BalanceSheet(Factor):
    """
    资产负债表
    """

    # 英文名
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
    def cname(self):
        return ['流动资产合计', '非流动资产合计', '资产总计', '流动负债合计', '非流动负债合计', '流动负债合计']

    def calculate(self, df_stocks):
        df_balancesheet = self.datasource.balance_sheet(self.stocks_info.stocks,
                                                        self.stocks_info.start_date,
                                                        self.stocks_info.end_date)
        return self._extract_fields(df_balancesheet)
