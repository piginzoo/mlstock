import math
import logging

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

    def __init__(self, cash, df_weekly, df_daily):
        self.cash = cash
        self.df_daily = df_daily
        self.df_weekly = df_weekly
        self.weekly_trade_dates = df_daily.trade_dates

        # 存储数据的结构
        self.positions = {}
        self.trades = []
        self.trade_history = []

    def distribute_cash(self):
        current_positions = 3
        available_positions = stock_num - current_positions
        cash4stock = math.ceil(cash / available_positions)
        return cash4stock

    def sell(self, trade, trade_date):
        df_stock = self.df_daily[(self.df_daily.trade_date == trade_date) &
                                 (self.df_daily.ts_code == trade.ts_code)]
        price = df_stock.low
        position = self.position[trade.ts_code]
        amount = price * position
        commission = amount * sell_commission_rate

        # 更新头寸
        self.cashin(amount - commission)
        # 更新仓位
        self.positions.pop(trade.ts_code, None)
        # 删除交易
        self.trades.pop(trade)
        # 保留交易历史
        trade.trade_date = trade_date
        self.trade_history.add(trade)

    def buy(self, trade, trade_date):
        df_stock = self.df_daily[(self.df_daily.trade_date == trade_date) &
                                 (self.df_daily.ts_code == trade.ts_code)]
        if len(df_stock) == 0:
            logger.warning("股票[%s]没有在[%s]无法交易，只能延后", trade.ts_code, trade_date)
            return

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
        # 删除交易
        self.trades.pop(trade)
        # 保留交易历史
        trade.trade_date = trade_date
        self.trade_history.add(trade)

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
        df_buy_stocks = self.df_weekly[self.df_weekly.trade_date == day_date]

        # 到调仓日，所有的买交易都取消了，但保留卖交易(没有卖出的要持续尝试卖出)
        self.clear_buy_trades()

        # 如果在
        for positon in self.position:
            if positon.ts_code in df_buy_stocks.ts_code: continue
            self.trades.add(Trade(positon.ts_code, day_date, 'sell'))

        for stock in df_buy_stocks:
            if self.is_in_position(stock): continue
            self.trades.add(Trade(stock.ts_code, day_date, 'buy'))

    def next(self):
        daily_trade_dates = self.df_daily.trade_dates
        for day_date in daily_trade_dates:

            if day_date in self.weekly_trade_dates:
                logger.debug("今日是调仓日：%s", day_date)
                self.handle_adjust_day(day_date)

            for trade in self.trades:
                if trade.action == "buy":
                    self.buy(trade, day_date)
                if trade.action == "sell":
                    self.sell(trade, day_date)
