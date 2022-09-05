import logging
import math
from datetime import datetime

import backtrader
import pandas as pd
from backtrader.feeds import PandasData

from mlstock.data.datasource import DataSource
from mlstock.utils import utils

logger = logging.getLogger(__name__)

PNL_LIMIT = 0.098
OPEN_GAP = 0.001


class MyPandasData(PandasData):
    def __init__(self, df):
        super().__init__()
        self.df = df


def is_limit_up(data):
    """
    判断是否涨停:
    1、比昨天的价格超过9.8%，不能为10%，太严格
    2、open、high、low和close的差距不超过价格的0.1% <-- No!No!No!不能用这个，这个就是未来函数了，想想你不可能知道第二天的close的
    对于买入的逻辑而言，你的做法应该是，
    如果你发现一直是高开，且最低价格一直不下来，直到快收盘，你就放弃购买了。
    那么，在回测的时候，复现这个逻辑，你的做法应该是，
        - 你如果发现当天的开盘价已经是昨日的9.8%了（不用管最高价了）
        - 最低价一直没和开盘价差距在0.1%（一直没拉开）
        这样，你就应该放弃买入了，视为这次机会放弃了。
    """
    pnl = (data.open[0] - data.close[-1]) / data.close[-1]
    gap = (data.open[0] - data.low[0]) / data.open[0]
    if pnl >= PNL_LIMIT and gap <= OPEN_GAP:
        logger.warning("今天是涨停日, 开盘涨幅[%.2f%%], 开盘和最低价相差比[%.2f%%]", pnl * 100, gap * 100)
        return True
    return False


def is_limit_low(data):
    """
    判断是否跌:
    1、比昨天的价格超过-9.8%，不能为10%，太严格
    对于卖出的逻辑而言，你的做法应该是：
        如果你发现一直是低开，最高价格一直上不来，直到快收盘，你就卖不出去了
    那么，在回测的时候，复现这个逻辑，你的做法应该是：
        - 你如果发现当天的开盘价已经超过昨日的-9.8%了（不用管最低了）
        - 最高价一直没和开盘价差距在0.1%（一直没拉开）
        这样，你就应该放弃买入了，视为这次机会放弃了。
    """
    pnl = (data.open[0] - data.close[-1]) / data.close[-1]
    gap = (data.high[0] - data.open[0]) / data.open[0]
    if pnl <= - PNL_LIMIT and gap <= OPEN_GAP:
        logger.warning("今天是跌停日, 开盘跌幅[%.2f%%], 最高价和开盘相差比[%.2f%%]", pnl * 100, gap * 100)
        return True
    return False


def get_trade_period(the_date, period, datasource):
    """
    返回某一天所在的周、月的交易日历中的开始和结束日期
    比如，我传入是 2022.2.15， 返回是的2022.2.2/2022.2.27（这2日是2月的开始和结束交易日）
    datasource是传入的
    the_date：格式是YYYYMMDD
    period：W 或 M
    """

    the_date = utils.str2date(the_date)

    # 读取交易日期
    df = datasource.trade_cal(exchange='SSE', start_date=today, end_date='20990101')
    # 只保存日期列
    df = pd.DataFrame(df, columns=['cal_date'])
    # 转成日期型
    df['cal_date'] = pd.to_datetime(df['cal_date'], format="%Y%m%d")
    # 如果今天不是交易日，就不需要生成
    if pd.Timestamp(the_date) not in df['cal_date'].unique(): return False

    # 把日期列设成index（因为index才可以用to_period函数）
    df = df[['cal_date']].set_index('cal_date')
    # 按照周、月的分组，对index进行分组
    df_group = df.groupby(df.index.to_period(period))
    # 看传入日期，是否是所在组的，最后一天，即，周最后一天，或者，月最后一天
    target_period = None
    for period, dates in df_group:
        if period.start_time < pd.Timestamp(the_date) < period.end_time:
            target_period = period
    if target_period is None:
        logger.warning("无法找到上个[%s]的开始、结束日期", period)
        return None, None
    return period[0], period[-1]


def is_trade_day():
    """
    判断是不是交易时间：9：30~11:30
    :return:
    """
    datasource = DataSource()
    trade_dates = list(datasource.trade_cal(start_date=utils.last_week(utils.today()), end_date=utils.today()))
    if utils.today() in trade_dates:
        return True
    return False


def next_trade_day(trade_date, df_calendar):
    """
    下一个交易日
    :return:
    """
    index = df_calendar[df_calendar == trade_date].index[0] + 1
    if index > len(df_calendar): return None
    return df_calendar[index]


def is_trade_time():
    FMT = '%H:%M:%S'
    now = datetime.strftime(datetime.now(), FMT)
    time_0930 = "09:30:00"
    time_1130 = "11:30:00"
    time_1300 = "13:00:00"
    time_1500 = "15:00:00"
    is_morning = time_0930 <= now <= time_1130
    is_afternoon = time_1300 <= now <= time_1500
    return is_morning or is_afternoon


def calc_size(broker, cash, data, price):
    """
    用来计算可以购买的股数：
    1、刨除手续费
    2、要是100的整数倍
    为了保守起见，用涨停价格来买，这样可能会少买一些。
    之前我用当天的close价格来算size，如果不打富余，第二天价格上涨一些，都会导致购买失败。
    """
    commission = broker.getcommissioninfo(data)
    commission = commission.p.commission
    # 注释掉了，头寸交给外面的头寸分配器（CashDistribute）来做了
    # cash = broker.get_cash()

    # 按照一个保守价格来买入
    size = math.ceil(cash * (1 - commission) / price)

    # 要是100的整数倍
    size = (size // 100) * 100
    return size


class LongOnlySizer(backtrader.Sizer):
    """
    自定义Sizer，经实践，不靠谱，
    他用的是下单当天的价格，即当前日期，不是下一个交易日，而是当前日，这个肯定不行了就。
    --------------------------------------------
    >[20170331] 信号出现：SMA5/10翻红,5/10多头排列,周MACD金叉,月KDJ金叉，买入
    >[20170331] 尝试买入：挂限价单[46.23]单，当前现金[126961.77]，买入股份[2700.00]份
    >[20170331] 计算买入股数: 价格[42.49],现金[126961.77],股数[2900.00]
        \__这个日期是错的，应该用20170405的价格
    >['20170405'] 交易失败，股票[000020.SZ]：'Margin'，现金[126961.77]
    """

    params = (('min_stake_unit', 100),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            commission = comminfo.p.commission
            price = data.open[0]
            size = math.ceil(cash * (1 - commission) / price)
            size = (size // 100) * 100
            logger.debug("[%s] 计算买入股数: 价格[%.2f],现金[%.2f],股数[%.2f]", self.strategy._date(), price, cash, size)
            return size


# python -m backtest.data_utils
if __name__ == '__main__':
    print("今天是交易日么？", is_trade_day())
