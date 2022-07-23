import logging
from abc import ABC, abstractmethod

from pandas import DataFrame

from mlstock.data.stock_info import StocksInfo

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
    def calculate(self, df: DataFrame):
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
