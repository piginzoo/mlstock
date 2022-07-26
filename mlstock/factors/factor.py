import logging
from abc import ABC, abstractmethod

import pandas as pd
from pandas import DataFrame

from mlstock.data.stock_info import StocksInfo
from mlstock.utils import utils

logger = logging.getLogger(__name__)


class Factor(ABC):
    """
    因子，也就是指标
    """

    def __init__(self, datasource, stocks_info: StocksInfo):
        self.datasource = datasource
        self.stocks_info = stocks_info

    # 英文名
    @property
    def name(self):
        return "Unknown"

    # 中文名
    @property
    def cname(self):
        return "未定义"

    @abstractmethod
    def calculate(self, stock_data):
        raise ImportError()

    @abstractmethod
    def merge(self, df_stocks, df_factor):
        raise ImportError()

    def _rename_finance_column_names(self, df: DataFrame):
        """
        - 把ann_date，改名成trade_date
        - 把其他的需要的字段的名字改成一个full_name，缩写看不懂
        """
        # 把tushare中的业务字段名改成full_name，便于理解
        df = self._rename(df, self.tushare_name, self.name)
        return df

    def _extract_fields(self, df: DataFrame):
        """
        - 把ts_code,ann_date,和其他需要的字段，剥离出来
        """
        # 注意，这里是trade_code，不是ann_date
        return df[['ts_code', 'trade_date'] + self.name]

    def _rename(self, df: DataFrame, _from: list, _to: list):
        name_pair = dict(zip(_from, _to))
        return df.rename(columns=name_pair)


class SimpleFactor(Factor):

    def merge(self, df_stocks, df_factor):
        """
        这个merge只是简单的把列合到一起，这里假设，df_factor包含了同df_stocks同样行数的数据，
        且，包含的列都是纯粹的特征列（不包含ts_code,trade_date等附加列）
        :param df_stocks:
        :param df_factor:
        :return:
        """
        assert len(df_factor) == len(df_factor)
        # index不一样没法对着行做列赋值，尽管行数一样，所以先都重置int的索引
        df_stocks = df_stocks.reset_index(drop=True)
        df_factor = df_factor.reset_index(drop=True)

        if type(df_factor)==pd.Series:
            df_factor.name = self.name
        if type(df_factor)==pd.DataFrame:
            df_factor.columns = self.name

        return pd.concat([df_stocks, df_factor], axis=1)


class ComplexMergeFactor(Factor):
    """合并数据和因子靠的是ts_code和trade_date做left join"""

    def merge(self, df_stocks, df_factor):
        # rename财务数据的公布日期ann_date=>trade_date
        df_factor.rename({'ann_date': 'trade_date'})
        # 做数据合并
        df_stocks = df_stocks.merge(df_factor, on=['ts_code', 'trade_date'], how='left')
        return df_stocks


class FinanceFactor(ComplexMergeFactor):

    def _numberic(self, df_finance):
        """
        由于tushare下载完后很多是nan列，或者，别错误定义成了text类型，这里要强制转成float
        self.name就是对应的财务指标列，是rename之后的
        """
        df_finance[self.name] = df_finance[self.name].astype('float')
        return df_finance

    @property
    def data_loader_func(self):
        raise NotImplementedError()

    def calculate(self, stock_data):
        df_weekly = stock_data.df_weekly
        """
        之所有要传入df_stocks，是因为要用他的日期，对每个日期进行TTM填充
        :param df_weekly: 股票周频数据
        :return:
        """
        # 由于财务数据，需要TTM，所以要溯源到1年前，所以要多加载前一年的数据
        start_date_last_year = utils.last_year(self.stocks_info.start_date)

        # 加载资产负债表数据
        df_finance = self.data_loader_func(self.stocks_info.stocks,
                                           start_date_last_year,
                                           self.stocks_info.end_date)

        # 把财务字段改成全名（tushare中的缩写很讨厌）
        df_finance = self._rename_finance_column_names(df_finance)

        # 把财务字段类型改成float，之前种种原因导致列是text类型的
        df_finance = self._numberic(df_finance)

        # 做财务数据的TTM处理
        df_finance = self.ttm(df_finance, self.name)

        # 按照股票的周频日期，来生成对应的指标（填充周频对应的财务指标）
        df_finance = self.fill(df_weekly, df_finance, self.name)

        # 只保留股票、日期和需要的特征列
        df_finance = self._extract_fields(df_finance)

        return df_finance
