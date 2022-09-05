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
        self.weekly_trade_dates = df_daily.trade_date.unique()

        # 存储数据的结构
        self.positions = {}
        self.trades = []
        self.trade_history = []
        self.df_values = DataFrame()

    def distribute_cash(self):
        current_positions = 3
        available_positions = stock_num - current_positions
        cash4stock = math.ceil(cash / available_positions)
        return cash4stock

    def sell(self, trade, trade_date):
        df_stock = self.df_daily[(self.df_daily.trade_date == trade_date) &
                                 (self.df_daily.ts_code == trade.ts_code)]
        if len(df_stock) == 0:
            logger.warning("股票[%s]没有在[%s]无数据，无法卖出，只能延后", trade.ts_code, trade_date)
            return False
        price = df_stock.low
        position = self.position[trade.ts_code]
        amount = price * position
        commission = amount * sell_commission_rate

        # 更新头寸
        self.cashin(amount - commission)
        # 更新仓位
        self.positions.pop(trade.ts_code, None) # None可以防止pop异常
        # 保留交易历史
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
        price = df_stock.high
        # 看看能分到多少钱
        cash4stock = self.distribute_cash()
        # 看看实际能卖多少手
        position = 100 * ((cash4stock / price) // 100)  # 最小单位是1手=100股
        # 计算实际费用
        actual_cost = position * price
        # 计算佣金
        commission = buy_commission_rate * actual_cost

        # 更新仓位
        self.positions[trade.ts_code] = position

        # 更新头寸
        self.cashout(actual_cost + commission)

        # 保留交易历史
        trade.trade_date = trade_date
        self.trade_history.append(trade)

        logger.debug("股票[%s]已于[%s]日按照最高价[%.2f]买入%d股,买入金额[%.2f],佣金[%.2f]",
                     trade.ts_code, trade_date, price, position, actual_cost, commission)
        return True

    def cashin(self, amount):
        self.cash += amount

    def cashout(self, amount):
        self.cash -= amount

    def is_in_position(self, ts_code):
        for position in self.positions:
            if position.ts_code == ts_code: return True
        return False

    def clear_buy_trades(self):
        self.trades = [t for t in self.trades if t.action == 'sell']

    def handle_adjust_day(self, day_date):
        """
        处理调仓日
        """
        df_buy_stocks = self.df_selected_stocks[self.df_selected_stocks.trade_date == day_date].ts_code

        # 到调仓日，所有的买交易都取消了，但保留卖交易(没有卖出的要持续尝试卖出)
        self.clear_buy_trades()

        # 如果在
        for positon in self.positions:
            if positon.ts_code in df_buy_stocks.ts_code:
                logger.info("待买股票[%s]已经在仓位中，无需卖出", positon.ts_code)
                continue
            self.trades.append(Trade(positon.ts_code, day_date, 'sell'))
            logger.debug("%s ，创建卖单，卖出持仓股票 [%s]", day_date, positon.ts_code)

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
        for ts_code, position in self.positions:

            df_stock = self.df_daily[self.df_daily.ts_code == ts_code]

            # TODO:如果停牌
            if len(df_stock) == 0:
                logger.warning(" %s 日没有股票 %s 的数据，当天它的市值计作 0 ", trade_date, ts_code)
                market_value = 0
            else:
                market_value = df_stock.close * position

            total_position_value += market_value

        total_value = total_position_value + self.cash
        self.df_values.append({'trade_date': trade_date,
                               'total_value': total_value,
                               'total_position_value': total_position_value,
                               'cash': self.cash}, ignore_index=True)
        logger.debug("更新 %s 日的市值 %.2f = %d只股票市值 %.2f + 持有的现金 %.2f",
                     trade_date, total_position_value, len(self.positions), total_position_value, self.cash)

    def execute(self):
        daily_trade_dates = self.df_daily.trade_date.unique()
        for day_date in daily_trade_dates:
            # import pdb;pdb.set_trace()

            if day_date in self.weekly_trade_dates:
                logger.debug("今日是调仓日：%s", day_date)
                self.handle_adjust_day(day_date)

            remove_flags = []
            for trade in self.trades:
                if trade.action == "buy":
                    remove_flags.append(self.buy(trade, day_date))
                elif trade.action == "sell":
                    remove_flags.append(self.sell(trade, day_date))
                raise ValueError(f"无效的交易类型：{trade.action}")

            # 保留那些失败的交易，等待明天重试
            original_position_size = len(self.positions)
            self.trades = [self.trades[i] for i, b in enumerate(remove_flags) if not b]
            logger.debug("%s 日后，仓位从%d=>%d只",original_position_size,len(self.positions))

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
    df_portfolio['pct_chg'] = df_portfolio.total_value.pct_change()
    df_portfolio['next_pct_chg'] = df_portfolio.pct_chg.shift(-1)
    df_portfolio[['cumulative_pct_chg', 'cumulative_pct_chg_baseline']] = \
        df_portfolio[['next_pct_chg', 'next_pct_chg_baseline']].apply(lambda x: (x + 1).cumprod() - 1)

    plot(df_portfolio, start_date, end_date, factor_names)

    # 计算各项指标
    metrics(df_portfolio)
