import math
import logging

from pandas import DataFrame

from mlstock.data.datasource import DataSource
from mlstock.ml.backtests import predict, select_top_n, plot
from mlstock.ml.backtests.metrics import metrics
from mlstock.utils.data_utils import next_trade_day

logger = logging.getLogger(__name__)

cash = 500000
buy_commission_rate = 0.00025 + 0.0002  # 券商佣金、过户费
sell_commission_rate = 0.00025 + 0.0002 + 0.001  # 券商佣金、过户费、印花税


class Trade:
    def __init__(self, ts_code, target_date, action):
        self.ts_code = ts_code
        self.target_date = target_date
        self.action = action
        self.actual_date = None


class Position:
    def __init__(self, ts_code, position, create_date, initial_value):
        self.ts_code = ts_code
        self.position = position
        self.create_date = create_date
        self.initial_value = initial_value


class Broker:

    def __init__(self, cash, df_selected_stocks, df_daily, df_calendar, conservative=False):
        self.cash = cash
        self.daily_trade_dates = df_daily.trade_date.unique()
        self.df_daily = df_daily.set_index(['trade_date', 'ts_code'])  # 设索引是为了加速，太慢了否则
        self.df_selected_stocks = df_selected_stocks
        self.weekly_trade_dates = df_selected_stocks.trade_date.unique()
        self.df_calendar = df_calendar
        self.conservative = conservative

        # 存储数据的结构
        self.positions = {}
        self.trades = []
        self.df_values = DataFrame()

    def distribute_cash(self):
        if self.cash <= 0:
            return None
        buy_stock_num = self.get_buy_trade_num()
        cash4stock = math.floor(self.cash / buy_stock_num)
        return cash4stock

    def sell(self, trade, trade_date):
        try:
            df_stock = self.df_daily.loc[(trade_date, trade.ts_code)]
        except KeyError:
            logger.warning("股票[%s]没有在[%s]无数据，无法卖出，只能延后", trade.ts_code, trade_date)
            return False

        # assert len(df_stock) == 1, f"根据{trade_date}和{trade.ts_code}筛选出多于1行的数据：{len(df_stock)}行"

        if self.conservative:
            price = df_stock.iloc[0].low
        else:
            price = df_stock.iloc[0].open
        position = self.positions[trade.ts_code]
        amount = price * position.position
        commission = amount * sell_commission_rate

        # 更新头寸,仓位,交易历史
        self.trades.remove(trade)
        self.cashin(amount - commission)
        self.positions.pop(trade.ts_code, None)  # None可以防止pop异常
        _return = (amount - position.initial_value) / position.initial_value

        trade.trade_date = trade_date

        logger.debug("股票[%s]在[%s]按最低价[%.2f]卖出,卖出金额[%.2f],佣金[%.2f],买入时价值[%.2f],收益[%.1f%%]",
                     trade.ts_code, trade_date, price, amount, commission, position.initial_value, _return * 100)
        return True

    def buy(self, trade, trade_date):
        # 使用try/exception + 索引loc是为了提速，直接用列，或者防止KeyError的intersection，都非常慢， 60ms vs 3ms，20倍关系
        # 另外，trade_date列是否是str还是date/int对速度影响不大
        # df_stock = self.df_daily.loc[self.df_daily.index.intersection([(trade_date, trade.ts_code)])]
        try:
            df_stock = self.df_daily.loc[(trade_date, trade.ts_code)]
        except KeyError:
            logger.warning("股票[%s]没有在[%s]无数据，无法买入，只能延后", trade.ts_code, trade_date)
            return False

        # assert len(df_stock) == 1, f"根据{trade_date}和{trade.ts_code}筛选出多于1行的数据：{len(df_stock)}行"

        # 保守取最高价
        if self.conservative:
            price = df_stock.iloc[0].high
        else:
            price = df_stock.iloc[0].open
        # 看看能分到多少钱
        cash4stock = self.distribute_cash()
        if cash4stock is None:
            logger.warning("现金[%.2f]为0，无法为股票[%s]分配资金了", self.cash, trade.ts_code)
            return False

        # 看看实际能卖多少手
        position = 100 * ((cash4stock / price) // 100)  # 最小单位是1手=100股
        if position == 0:
            logger.warning("资金分配失败：从总现金[%.2f]中分配[%.2f]给股票[%s]（价格%.2f）失败",
                           self.cash, cash4stock, trade.ts_code, price)
            return False

        # 计算实际费用
        actual_cost = position * price
        # 计算佣金
        commission = buy_commission_rate * actual_cost

        # 更新仓位,头寸,交易历史
        self.trades.remove(trade)
        self.positions[trade.ts_code] = Position(trade.ts_code, position, trade_date, actual_cost)
        self.cashout(actual_cost + commission)

        logger.debug("股票[%s]已于[%s]日按照最高价[%.2f]买入%d股,买入金额[%.2f],佣金[%.2f]",
                     trade.ts_code, trade_date, price, position, actual_cost, commission)
        return True

    def cashin(self, amount):
        old = self.cash
        self.cash += amount
        logger.debug("现金增加：%2.f=>%.2f", old, self.cash)

    def cashout(self, amount):
        old = self.cash
        self.cash -= amount
        logger.debug("现金减少：%2.f=>%.2f", old, self.cash)

    def is_in_position(self, ts_code):
        for position_ts_code, _ in self.positions.items():
            if position_ts_code == ts_code: return True
        return False

    def clear_buy_trades(self):
        self.trades = [t for t in self.trades if t.action == 'sell']

    def is_in_sell_trades(self, ts_code):
        for t in self.trades:
            if t.action != 'sell': continue
            if t.ts_code == ts_code: return True
        return False

    def get_buy_trade_num(self):
        return len([t for t in self.trades if t.action == 'buy'])

    def handle_adjust_day(self, day_date):
        """
        处理调仓日
        """
        df_buy_stocks = self.df_selected_stocks[self.df_selected_stocks.trade_date == day_date].ts_code

        # 到调仓日，所有的买交易都取消了，但保留卖交易(没有卖出的要持续尝试卖出)
        self.clear_buy_trades()

        next_trade_date = next_trade_day(day_date, self.df_calendar)
        if next_trade_date is None:
            logger.warning("无法获得[%s]的下一个交易日,不做任何调仓", day_date)
            return

        logger.debug("调仓日[%s]模型建议买入%d只股票，清仓%d只股票", day_date, len(df_buy_stocks), len(self.positions))

        for ts_code, position in self.positions.items():
            if ts_code in df_buy_stocks.unique():
                logger.info("待清仓股票[%s]在本周购买列表中，无需卖出", ts_code)
                continue
            if self.is_in_sell_trades(ts_code):
                logger.warning("股票[%s]已经在卖单中，可能是还未卖出，无需再创建卖单了", ts_code)
                continue
            self.trades.append(Trade(ts_code, next_trade_date, 'sell'))
            logger.debug("%s ，创建下个交易日[%s]卖单，卖出持仓股票 [%s]", day_date, next_trade_date, ts_code)

        for stock in df_buy_stocks.unique():
            if self.is_in_position(stock):
                logger.info("待买股票[%s]已经在仓位中，无需买入", stock)
                continue
            self.trades.append(Trade(stock, next_trade_date, 'buy'))
            logger.debug("%s ，创建下个交易日[%s]买单，买入股票 [%s]", day_date, next_trade_date, stock)

    def update_market_value(self, trade_date):
        """
        # 日子，总市值，现金，市值
        市值 = sum(position_i * price_i)
        """
        total_position_value = 0
        for ts_code, position in self.positions.items():
            logger.debug("查找股票[%s] %s数据", ts_code, trade_date)

            try:
                df_the_stock = self.df_daily.loc[(trade_date, ts_code)]
                # assert len(df_the_stock) == 1, f"根据{trade_date}和{ts_code}筛选出多于1行的数据：{len(df_the_stock)}行"
                market_value = df_the_stock.iloc[0].close * position.position
            except KeyError:
                logger.warning(" %s 日没有股票 %s 的数据，当天它的市值计作 0 ", trade_date, ts_code)
                market_value = 0

            total_position_value += market_value

        total_value = total_position_value + self.cash
        self.df_values = self.df_values.append({'trade_date': trade_date,
                                                'total_value': total_value,
                                                'total_position_value': total_position_value,
                                                'cash': self.cash}, ignore_index=True)
        logger.debug("%s 市值 %.2f = %d只股票市值 %.2f + 持有现金 %.2f",
                     trade_date, total_value, len(self.positions), total_position_value, self.cash)

    def execute(self):
        for day_date in self.daily_trade_dates:
            original_position_size = len(self.positions)

            if day_date in self.weekly_trade_dates:
                logger.debug(" ================ 调仓日：%s ================", day_date)
                self.handle_adjust_day(day_date)

            is_transaction_succeeded = []
            for trade in self.trades:
                # 只有大于和等于交易的目标日才交易（下单是在第二天才执行）
                if trade.target_date < day_date: continue
                if trade.action == "buy":
                    is_transaction_succeeded.append(self.buy(trade, day_date))
                    continue
                elif trade.action == "sell":
                    is_transaction_succeeded.append(self.sell(trade, day_date))
                    continue
                raise ValueError(f"无效的交易类型：{trade.action}")

            # 先卖
            sell_trades = [x for x in self.trades if x.action == 'sell']
            for trade in sell_trades:
                self.sell(trade, day_date)

            # 后买
            buy_trades = [x for x in self.trades if x.action == 'buy']
            for trade in buy_trades:
                self.buy(trade, day_date)

            if original_position_size != len(self.positions):
                logger.debug("%s 日后，仓位变化，从%d=>%d 只", day_date, original_position_size, len(self.positions))

            self.update_market_value(day_date)


def main(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names):
    """
    先预测出所有的下周收益率、下周涨跌 => df_data，
    然后选出每周的top30 => df_selected_stocks，
    然后使用Broker，来遍历每天的交易，每周进行调仓，并，记录下每周的股票+现价合计价值 => df_portfolio
    最后计算出next_pct_chg、cumulative_pct_chg，并画出plot，计算metrics
    """
    datasource = DataSource()

    df_data = predict(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names)
    df_limit = datasource.limit_list()
    df_index = datasource.index_weekly('000001.SH',start_date,end_date)
    ts_codes = df_data.ts_code.unique().tolist()
    df_daily = datasource.daily(ts_codes, start_date, end_date, adjust='')
    df_daily = df_daily.sort_values(['trade_date', 'ts_code'])

    df_selected_stocks = select_top_n(df_data, df_limit)
    df_calendar = datasource.trade_cal(start_date, end_date)

    broker = Broker(cash, df_selected_stocks, df_daily, df_calendar)
    broker.execute()
    df_portfolio = broker.df_values
    df_portfolio.sort_values('trade_date')

    # 把基准收益拼接进去
    df_baseline = df_data[['trade_date', 'next_pct_chg_baseline']].drop_duplicates()
    # 只筛出来周频的市值来
    df_portfolio = df_baseline.merge(df_portfolio, how='left', on='trade_date')
    # 拼接上指数
    df_index = df_index['trade_date','close']
    df_index.columns = ['trade_date','index']
    df_portfolio = df_portfolio.merge(df_index, how='left', on='trade_date')

    # 准备pct、next_pct_chg、和cumulative_xxxx
    df_portfolio['pct_chg'] = df_portfolio.total_value.pct_change()
    df_portfolio['next_pct_chg'] = df_portfolio.pct_chg.shift(-1)
    df_portfolio[['cumulative_pct_chg', 'cumulative_pct_chg_baseline']] = \
        df_portfolio[['next_pct_chg', 'next_pct_chg_baseline']].apply(lambda x: (x + 1).cumprod() - 1)


    # 组合的收益率情况
    df_portfolio = df_selected_stocks.groupby('trade_date')[['next_pct_chg', 'next_pct_chg_baseline']].mean().reset_index()
    df_portfolio.columns = ['trade_date', 'next_pct_chg', 'next_pct_chg_baseline']
    df_portfolio[['cumulative_pct_chg', 'cumulative_pct_chg_baseline']] = \
        df_portfolio[['next_pct_chg', 'next_pct_chg_baseline']].apply(lambda x: (x + 1).cumprod() - 1)


    df_portfolio = df_portfolio[~df_portfolio.cumulative_pct_chg.isna()]

    plot(df_portfolio, start_date, end_date, factor_names)

    # 计算各项指标
    metrics(df_portfolio)
