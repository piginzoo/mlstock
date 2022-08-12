"""
盈利收益率
"""
import logging
from mlstock.factors.factor import ComplexMergeFactor
from mlstock.utils import utils

logger = logging.getLogger(__name__)

"""
EP = 净利润（TTM）/总市值
盈利收益率 EP（Earn/Price） = 盈利/价格
其实，就是1/PE（市盈率），
这里，就说说PE，因为EP就是他的倒数：
PE = PRICE / EARNING PER SHARE，指股票的本益比，也称为“利润收益率”。 
本益比是某种股票普通股每股市价与每股盈利的比率，所以它也称为股价收益比率或市价盈利比率。
- [基本知识解读 -- PE, PB, ROE，盈利收益率](https://xueqiu.com/4522742712/61623733)

"""


class DailyIndicator(ComplexMergeFactor):
    """
    daily_basic 中提供了3个指标
    """

    # 英文名
    @property
    def name(self):
        return ["EP", "SP", "BP"]

    # 中文名
    @property
    def cname(self):
        return ["净收益(TTM)/总市值", "营业收入（TTM）/总市值", "净资产/总市值"]

    def calculate(self, stock_data):
        df_daily_basic = stock_data.df_daily_basic
        df_daily_basic = df_daily_basic.sort_values(['ts_code', 'trade_date'])
        # 如果缺失，用这天之前的数据来填充（ffill)
        # 这样做不行，ts_code和trade_date两列丢了，pandas1.3.4也不行，只好逐个fill了
        # df_daily_basic = df_daily_basic.groupby(by=['ts_code']).fillna(method='ffill').reset_index()
        df_daily_basic[['pe_ttm', 'ps_ttm', 'pb']] = df_daily_basic.groupby('ts_code').ffill().bfill()[['pe_ttm', 'ps_ttm', 'pb']]
        df_daily_basic[self.name[0]] = 1 / df_daily_basic['pe_ttm']  # EP = 1/PE（市盈率）
        df_daily_basic[self.name[1]] = 1 / df_daily_basic['ps_ttm']  # SP = 1/PS（市销率）
        df_daily_basic[self.name[2]] = 1 / df_daily_basic['pb']  # SP = 1/PS（市销率）
        return df_daily_basic[['ts_code', 'trade_date'] + self.name]


# python -m mlstock.factors.daily_indicator
if __name__ == '__main__':
    from mlstock.data import data_loader
    from mlstock.data.datasource import DataSource
    from mlstock.data.stock_info import StocksInfo

    utils.init_logger(file=False)

    start_date = "20180101"
    end_date = "20200101"
    stocks = ['000019.SZ', '000063.SZ', '000068.SZ', '000422.SZ']
    datasource = DataSource()
    stocks_info = StocksInfo(stocks, start_date, end_date)
    stock_data = data_loader.load(datasource, stocks, start_date, end_date)
    df = DailyIndicator(datasource, stocks_info).calculate(stock_data)
    print(df)
    print("NA缺失比例", df.isna().sum() / len(df))
