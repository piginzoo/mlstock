import math
import logging

from pandas import DataFrame

from mlstock.data.datasource import DataSource
from mlstock.ml.backtests import predict, select_top_n, plot
from mlstock.ml.backtests.metrics import metrics

logger = logging.getLogger(__name__)

stock_num = 30
cash = 500000
buy_commission_rate = 0.00025 + 0.0002  # 券商佣金、过户费
sell_commission_rate = 0.00025 + 0.0002 + 0.001  # 券商佣金、过户费、印花税


class Trade:
    def __init__(self, ts_code, create_date, action):
        self.ts_code = ts_code
        self.create_date = create_date
        self.action = action
        self.trade_date = None


class Position:
    def __init__(self, ts_code, trade_date):
        self.ts_code = ts_code
        self.trade_date = trade_date


class Broker:

    def __init__(self, cash, df_selected_stocks, df_daily):
        self.cash = cash
        self.df_daily = df_daily
        self.df_selected_stocks = df_selected_stocks
        self.weekly_trade_dates = df_selected_stocks.trade_date.unique()

        # 存储数据的结构
        self.positions = {}
        self.trades = []
        self.trade_history = []
        self.df_values = DataFrame()

    def distribute_cash(self):
        if self.cash <= 0:
            return None
        buy_stock_num = self.get_buy_trade_num()
        cash4stock = math.floor(self.cash / buy_stock_num)
        return cash4stock

    def sell(self, trade, trade_date):
        df_stock = self.df_daily[(self.df_daily.trade_date == trade_date) &
                                 (self.df_daily.ts_code == trade.ts_code)]
        if len(df_stock) == 0:
            logger.warning("股票[%s]没有在[%s]无数据，无法卖出，只能延后", trade.ts_code, trade_date)
            return False

        assert len(df_stock) == 1
        price = df_stock.iloc[0].low
        position = self.positions[trade.ts_code]
        amount = price * position
        commission = amount * sell_commission_rate

        # 更新头寸,仓位,交易历史
        self.trades.pop(trade)
        self.cashin(amount - commission)
        self.positions.pop(trade.ts_code, None)  # None可以防止pop异常
        trade.trade_date = trade_date
        self.trade_history.append(trade)

        logger.debug("股票[%s]已于[%s]日按照最低价[%.2f]被卖出,卖出金额[%.2f],佣金[%.2f]",
                     trade.ts_code, trade_date, price, amount, commission)
        return True

    def buy(self, trade, trade_date):
        df_stock = self.df_daily[(self.df_daily.trade_date == trade_date) &
                                 (self.df_daily.ts_code == trade.ts_code)]
        if len(df_stock) == 0:
            logger.warning("股票[%s]没有在[%s]无数据，无法买入，只能延后", trade.ts_code, trade_date)
            return False

        # 保守取最高价
        assert len(df_stock) == 1
        price = df_stock.iloc[0].high
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
        self.trades.pop(trade)
        self.positions[trade.ts_code] = position
        self.cashout(actual_cost + commission)
        trade.trade_date = trade_date
        self.trade_history.append(trade)

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

        # 如果在
        if len(self.positions) > 0:
            logger.debug("仓位中有%d只股票，需要清仓", len(self.positions))
        for ts_code, position in self.positions.items():
            if ts_code in df_buy_stocks:
                logger.info("待买股票[%s]已经在仓位中，无需卖出", ts_code)
                continue
            if self.is_in_sell_trades(ts_code):
                logger.warning("股票[%s]已经在卖单中，可能是还未卖出，无需再创建卖单了", ts_code)
            else:
                self.trades.append(Trade(ts_code, day_date, 'sell'))
                logger.debug("%s ，创建卖单，卖出持仓股票 [%s]", day_date, ts_code)

        if len(df_buy_stocks) > 0:
            logger.debug("模型预测的有%d只股票，需要买入", len(df_buy_stocks))
        for stock in df_buy_stocks:
            if self.is_in_position(stock):
                logger.info("待买股票[%s]已经在仓位中，无需买入", stock)
                continue
            self.trades.append(Trade(stock, day_date, 'buy'))
            logger.debug("%s ，创建买单，买入股票 [%s]", day_date, stock)

    def record_value(self, trade_date):
        """
        # 日子，总市值，现金，市值
        市值 = sum(position_i * price_i)
        """
        total_position_value = 0
        for ts_code, position in self.positions.items():

            df_the_stock = self.df_daily[(self.df_daily.ts_code == ts_code) & (self.df_daily.trade_date == trade_date)]

            # TODO:如果停牌
            if len(df_the_stock) == 0:
                logger.warning(" %s 日没有股票 %s 的数据，当天它的市值计作 0 ", trade_date, ts_code)
                market_value = 0
            else:
                assert len(df_the_stock) == 1
                market_value = df_the_stock.iloc[0].close * position

            total_position_value += market_value

        total_value = total_position_value + self.cash
        self.df_values = self.df_values.append({'trade_date': trade_date,
                                                'total_value': total_value,
                                                'total_position_value': total_position_value,
                                                'cash': self.cash}, ignore_index=True)
        logger.debug("更新 %s 日的市值 %.2f = %d只股票市值 %.2f + 持有的现金 %.2f",
                     trade_date, total_value, len(self.positions), total_position_value, self.cash)

    def execute(self):
        daily_trade_dates = self.df_daily.trade_date.unique()
        for day_date in daily_trade_dates:
            original_position_size = len(self.positions)

            if day_date in self.weekly_trade_dates:
                logger.debug("今日是调仓日：%s", day_date)
                self.handle_adjust_day(day_date)

            is_transaction_succeeded = []
            for trade in self.trades:
                if trade.action == "buy":
                    is_transaction_succeeded.append(self.buy(trade, day_date))
                    continue
                elif trade.action == "sell":
                    is_transaction_succeeded.append(self.sell(trade, day_date))
                    continue
                raise ValueError(f"无效的交易类型：{trade.action}")

            buy_trades = [x for x in self.trades if x.action == 'buy']
            for trade in buy_trades:
                self.buy(trade)

            sell_trades = [x for x in self.trades if x.action == 'sell']

            # 保留那些失败的交易，等待明天重试
            self.trades = [self.trades[i] for i, b in enumerate(is_transaction_succeeded) if not b]
            if original_position_size != len(self.positions):
                logger.debug("%s 日后，仓位变化，从%d=>%d 只", day_date, original_position_size, len(self.positions))

            self.record_value(day_date)


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
    ts_codes = df_data.ts_code.unique().tolist()
    df_daily = datasource.daily(ts_codes, start_date, end_date)

    df_selected_stocks = select_top_n(df_data, df_limit)

    broker = Broker(cash, df_selected_stocks, df_daily)
    broker.execute()
    df_portfolio = broker.df_values
    df_portfolio.sort_values('trade_date')

    # 把基准收益拼接进去
    df_baseline = df_data[['trade_date', 'next_pct_chg_baseline']].drop_duplicates()
    # 只筛出来周频的市值来
    df_portfolio = df_baseline.merge(df_portfolio, how='left', on='trade_date')

    # 准备pct、next_pct_chg、和cumulative_xxxx
    df_portfolio['pct_chg'] = df_portfolio.total_value.pct_change()
    df_portfolio['next_pct_chg'] = df_portfolio.pct_chg.shift(-1)
    df_portfolio[['cumulative_pct_chg', 'cumulative_pct_chg_baseline']] = \
        df_portfolio[['next_pct_chg', 'next_pct_chg_baseline']].apply(lambda x: (x + 1).cumprod() - 1)

    df_portfolio = df_portfolio[~df_portfolio.cumulative_pct_chg.isna()]

    plot(df_portfolio, start_date, end_date, factor_names)

    # 计算各项指标
    metrics(df_portfolio)
