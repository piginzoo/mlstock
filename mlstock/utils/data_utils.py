import logging
import math
import backtrader
import pandas as pd
from backtrader.feeds import PandasData

from datasource import datasource_factory
from utils import utils
from datetime import datetime

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
    datasource = datasource_factory.create("tushare")
    trade_dates = list(datasource.trade_cal(start_date=utils.last_week(utils.today()), end_date=utils.today()))
    if utils.today() in trade_dates:
        return True
    return False


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


def handle_finance_ttm(stock_codes,
                       df_finance,
                       trade_dates,
                       col_name_value,
                       col_name_finance_date='end_date'):
    """
    处理TTM：以当天为基准，向前滚动12个月的数据，
    用于处理类ROE_TTM数据，当然不限于ROE，只要是同样逻辑的都支持。

    @:param finance_date  - 真正的财报定义的日期，如3.30、6.30、9.30、12.31

    ts_code    ann_date  end_date      roe
    600000.SH  20201031  20200930   7.9413
    600000.SH  20200829  20200630   5.1763
    600000.SH  20200425  20200331   3.0746
    600000.SH  20200425  20191231  11.4901
    600000.SH  20191030  20190930   9.5587 <----- 2019.8.1日可回溯到的日期
    600000.SH  20190824  20190630   6.6587
    600000.SH  20190430  20190331   3.4284
    600000.SH  20190326  20181231  12.4674

    处理方法：
    比如我要填充每一天的ROE_TTM，就要根据当前的日期，回溯到一个可用的ann_date（发布日期），
    然后以这个日期，作为可用数据，计算ROE_TTM。
    比如当前日是2019.8.1日，回溯到2019.10.30(ann_date)日发布的3季报（end_date=20190930, 0930结尾为3季报），
    然后，我们的计算方法就是，用3季报，加上去年的年报，减去去年的3季报。
    -----
    所以，我们抽象一下，所有的规则如下：
    - 如果是回溯到年报，直接用年报作为TTM
    - 如果回溯到1季报、半年报、3季报，就用其 + 去年的年报 - 去年起对应的xxx报的数据，这样粗暴的公式，是为了简单
    """

    # 提取，发布日期，股票，财务日期，财务指标 ，4列
    df_finance = df_finance[['datetime', 'code', col_name_finance_date, col_name_value]]
    # 剔除Nan
    df_finance.dropna(inplace=True)

    # 对时间，升序排列
    df_finance.sort_values('datetime', inplace=True)

    # 未来的，ttm列名
    ttm_col_name_value = col_name_value + "_ttm"

    # 创建空的结果DataFrame
    df_factor = pd.DataFrame(columns=['datetime', 'code', col_name_value])

    # 返回的数据，应该是交易日数据；一只一只股票的处理
    for stock_code in stock_codes:

        # 过滤一只股票
        df_stock_finance = df_finance[df_finance['code'] == stock_code]

        logger.debug("处理股票[%s]财务数据%d条", stock_code, len(df_stock_finance))

        # 处理每一天
        for the_date in trade_dates:

            # 找到最后发布的行：按照当前日作为最后一天，去反向搜索发布日在当前日之前的数据，取最后一条，就是最后发布的数据
            series_last_one = df_stock_finance[df_stock_finance['datetime'] <= the_date].iloc[-1]

            # 取出最后发布的财务日期
            finance_date = series_last_one[col_name_finance_date]

            # 取出最后发布的财务日期对应的指标值
            current_period_value = series_last_one[col_name_value]

            # 如果这条财务数据是年报数据
            if finance_date.endswith("1231"):
                # 直接用这条数据了
                value = current_period_value
                # logger.debug("财务日[%s]是年报数据，使用年报指标[%.2f]作为当日指标", finance_date, value)
            else:
                # 如果回溯到1季报、半年报、3季报，就用其 + 去年的年报 - 去年起对应的xxx报的数据，这样粗暴的公式，是为了简单
                last_year_value = __last_year_value(df_stock_finance, col_name_finance_date, col_name_value,
                                                    finance_date)
                last_year_same_period_value = __last_year_period_value(df_stock_finance,
                                                                       col_name_finance_date,
                                                                       col_name_value,
                                                                       finance_date)
                # 如果去年年报数据为空，或者，也找不到去年的同期的数据，
                if last_year_value is None or last_year_same_period_value is None:
                    value = __calculate_ttm_by_peirod(current_period_value, finance_date)
                    # logger.debug("财务日[%s]是非年报数据，无去年报指标，使用N倍当前指标[%.2f]作为当日指标", finance_date, value)
                else:
                    # 当日指标 = 今年同期 + 年报指标 - 去年同期
                    value = current_period_value + last_year_value - last_year_same_period_value
                    # logger.debug("财务日[%s]是非年报数据，今年同期[%.2f]+年报指标[%.2f]-去年同期[%.2f]=[%.2f]作为当日指标",
                    #              finance_date,
                    #              current_period_value,
                    #              last_year_value,
                    #              last_year_same_period_value,
                    #              value)

            df_factor = df_factor.append(
                {'datetime': the_date,
                 'code': stock_code,
                 col_name_value: current_period_value,
                 ttm_col_name_value: value},
                ignore_index=True)
    logger.debug("生成%d条TTM数据", len(df_factor))
    return df_factor




def handle_finance_fill(datasource,
                        stock_codes,
                        start_date,
                        end_date,
                        finance_index_col_name_value):
    """
    处理财务数据填充，因为财政指标只在年报发布时候提供，所以要填充那些非发布日的数据，
    比如财务数据仅提供了财务报表发表日的的数据，那么我们需要用这个数据去填充其他日子，
    填充原则是，以发布日为基准，当日数据以最后发布日的数据为准，
    算法是用通用日历来填充其他数据，但是，可能某天此股票停盘，无所谓，还是给他算出来，
    实现是，按照日历创建空记录集，然后做左连接，空位用前面的数据补齐
    有个细节，开始的日子需要再之前的财务数据，因此，我只好多query1年前的财务数据来处理，最终在过滤掉之前的数据
    """

    # 需要把提前1年的财务数据和日历都得到
    start_date_1years_ago = utils.last_year(start_date, num=1)
    # 交易日期（包含1年前）
    trade_dates = datasource.trade_cal(start_date_1years_ago, end_date)
    # 财务数据（包含1年前的）
    df_finance = datasource.fina_indicator(stock_codes, start_date_1years_ago, end_date)
    # 提取，发布日期，股票，财务日期，财务指标 ，4列
    df_finance = df_finance[['code', 'datetime', finance_index_col_name_value]]
    # 对时间，升序排列
    df_finance.sort_values('datetime', inplace=True)
    # 创建每个交易日为一行的一个辅助dataframe，用于生成每个股票的交易数据
    df_calender = pd.DataFrame(trade_dates)
    df_calender.columns = ['datetime']
    # 创建空的结果DataFrame，保存最终结果
    df_result = pd.DataFrame(trade_dates, columns=['code', 'datetime', finance_index_col_name_value])
    # 返回的数据，应该是交易日数据；一只一只股票的处理
    for stock_code in stock_codes:
        # 过滤一只股票
        df_stock_finance = df_finance[df_finance['code'] == stock_code]
        logger.debug("处理股票[%s]财务数据%d条", stock_code, len(df_stock_finance))
        # 左连接，交易日（左连接）财务数据，这样，没有的交易日，数据为NAN
        df_join = df_calender.merge(df_stock_finance, how="left", on='datetime')
        # 为防止日期顺序有问题，重新排序
        df_join = df_join.sort_values('datetime')
        # 向下填充nan值，这个是一个神奇的方法，跟Stack Overflow上学的
        # 参考 https://stackoverflow.com/questions/27905295/how-to-replace-nans-by-preceding-or-next-values-in-pandas-dataframe
        df_join = df_join.fillna(method='ffill')
        # 补齐股票代码
        df_join['code'] = stock_code
        # 因为提前了1年的数据，所以，要把这些提前数据过滤掉
        df_join = df_join[df_join.datetime >= start_date]

        # 做一个断言，理论上不应该有nan数据
        nan_sum = df_join[finance_index_col_name_value].isnull().sum()
        assert nan_sum == 0, f"你需要多传一年的财务数据，防止NAN: {nan_sum}行NAN "

        # 合并到总结果中
        df_result = df_result.append(df_join, ignore_index=True)

    return df_result


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
