import logging

from mlstock import const
from mlstock.factors.factor import ComplexMergeFactor
from mlstock.factors.mixin.fill_mixin import FillMixin
from mlstock.utils import utils

logger = logging.getLogger(__name__)


class StakeHolder(ComplexMergeFactor, FillMixin):
    """
    股东变化率
    """

    @property
    def name(self):
        return 'stake_holder_chg'

    @property
    def cname(self):
        return '股东人数'

    def calculate(self, stock_data):
        df_weekly = stock_data.df_weekly

        # 因为需要回溯，所以用了同一回溯标准（类MACD指标）
        start_date = utils.last_week(self.stocks_info.start_date, const.RESERVED_PERIODS)

        df_stake_holder = self.datasource.stock_holder_number(self.stocks_info.stocks,
                                                              start_date,
                                                              self.stocks_info.end_date)
        df_stake_holder = df_stake_holder.sort_values(by='ann_date')
        df = self.fill(df_weekly, df_stake_holder, 'holder_num')
        df[self.name] = (df.holder_num - df.holder_num.shift(1)) / df.holder_num.shift(1)
        return df[['ts_code', 'trade_date', self.name]]


# python -m mlstock.factors.stake_holder
if __name__ == '__main__':
    from mlstock.data import data_loader
    from mlstock.data.datasource import DataSource
    from mlstock.data.stock_info import StocksInfo
    import pandas

    pandas.set_option('display.max_rows', 1000000)

    utils.init_logger(file=False)

    datasource = DataSource()
    start_date = "20180101"
    end_date = "20200101"
    stocks = ['000010.SZ', '000014.SZ']
    stocks_info = StocksInfo(stocks, start_date, end_date)

    df_stocks = data_loader.load(datasource, stocks, start_date, end_date)

    df = StakeHolder(datasource, stocks_info).calculate(df_stocks)
    print(df)
