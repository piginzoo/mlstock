import logging

from mlstock.factors.factor import ComplexMergeFactor
from mlstock.factors.mixin.fill_mixin import FillMixin
from mlstock.utils import utils

logger = logging.getLogger(__name__)


class StakeHolder(ComplexMergeFactor, FillMixin):

    @property
    def name(self):
        return 'stake_holder'

    @property
    def cname(self):
        return '股东人数'

    def calculate(self, stock_data):
        df_weekly = stock_data.df_weekly
        df_stake_holder = self.datasource.stock_holder_number(self.stocks_info.stocks,
                                                              self.stocks_info.start_date,
                                                              self.stocks_info.end_date)
        df = self.fill(df_weekly, df_stake_holder, 'holder_num')
        df = df.rename(columns={'holder_num': self.name})
        return df[['ts_code', 'trade_date', self.name]]


# python -m mlstock.factors.stake_holder
if __name__ == '__main__':
    from mlstock.data import data_loader
    from mlstock.data.datasource import DataSource
    from mlstock.data.stock_info import StocksInfo

    utils.init_logger(file=False)

    datasource = DataSource()
    start_date = '20170703'
    end_date = '20190826'
    stocks = ['600000.SH', '002357.SZ', '000404.SZ', '600230.SH']
    stocks_info = StocksInfo(stocks, start_date, end_date)

    df_stocks = data_loader.load(datasource, stocks, start_date, end_date)

    df = StakeHolder(datasource, stocks_info).calculate(df_stocks)
    print(df)
