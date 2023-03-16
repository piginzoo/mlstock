"""
这个使用B站UP主的数据源导入后的mysql进行访问
"""
import logging
import time

import pandas as pd

from mlstock import const
from mlstock.utils import db_utils, utils


logger = logging.getLogger(__name__)


class DataSource:
    def __init__(self,conf=None):
        if conf is None:
            conf = utils.load_config()
        self.db_engine = db_utils.connect_db(conf)

    def daily(self, stock_code, start_date=None, end_date=None, adjust='hfq'):
        if not start_date: start_date = const.EALIEST_DATE
        if not end_date: end_date = utils.today()

        if adjust == None or adjust == '':
            table_name = "daily"
        else:
            table_name = f"daily_{adjust}"

        if type(stock_code) == list:
            stock_codes = db_utils.list_to_sql_format(stock_code)
            start_time = time.time()

            df_all = pd.read_sql(
                f'select * from {table_name} where ts_code in ({stock_codes}) and trade_date>="{start_date}" and trade_date<="{end_date}"',
                self.db_engine)

            # df_all = None
            # start_time = time.time()
            # for i, stock in enumerate(stock_code):
            #     df_daily = self.__daliy_one(stock, start_date, end_date, adjust)
            #     if df_all is None:
            #         df_all = df_daily
            #     else:
            #         df_all = df_all.append(df_daily)

            logger.debug("获取 %s ~ %s %d 只股票的交易日数据：%d 条, 耗时 %.2f 秒",
                         start_date, end_date, len(stock_code), len(df_all), time.time() - start_time)
            return df_all
        else:
            # df_one = self.__daliy_one(stock_code, start_date, end_date, adjust)
            df_one = pd.read_sql(
                f'select * from {table_name} where ts_code="{stock_code}" and trade_date>="{start_date}" and trade_date<="{end_date}"',
                self.db_engine)

            logger.debug("获取 %s ~ %s 股票[%s]的交易数据：%d 条", start_date, end_date, stock_code, len(df_one))
            return df_one

    def weekly(self, stock_code, start_date, end_date):
        df = pd.read_sql(
            f'select * from weekly_hfq where ts_code="{stock_code}" and trade_date>="{start_date}" and trade_date<="{end_date}"',
            self.db_engine)
        return df

    def monthly(self, stock_code, start_date, end_date):
        df = pd.read_sql(
            f'select * from monthly_hfq where ts_code="{stock_code}" and trade_date>="{start_date}" and trade_date<="{end_date}"',
            self.db_engine)
        return df

    def daily_basic(self, stock_code, start_date, end_date):
        assert type(stock_code) == list or type(stock_code) == str, type(stock_code)
        start_time = time.time()
        if type(stock_code) == list:
            df_basics = [self.__daily_basic_one(stock, start_date, end_date) for stock in stock_code]
            logger.debug("获取%d只股票的每日基本信息数据%d条，耗时 : %.2f秒", len(stock_code), len(df_basics), time.time() - start_time)
            # print(df_basics)
            return pd.concat(df_basics)
        if stock_code is None or stock_code == '':
            """返回2个日期间的所有股票信息"""
            df = pd.read_sql(
                f'select * from daily_basic \
                    where trade_date>="{start_date}" and trade_date<="{end_date}"',
                self.db_engine)
            logger.debug("获取从%s~%s之间的所有股票基本信息数据%d条，耗时 : %.2f秒", start_date, end_date, len(df), time.time() - start_time)
            return df

        return self.__daily_basic_one(stock_code, start_date, end_date)

    def __daily_basic_one(self, stock_code, start_date, end_date):
        """返回每日的其他信息，主要是市值啥的"""
        df = pd.read_sql(
            f'select * from daily_basic \
                where ts_code="{stock_code}" and trade_date>="{start_date}" and trade_date<="{end_date}"',
            self.db_engine)
        return df

    # 指数日线行情
    def index_daily(self, index_code, start_date, end_date):
        df = pd.read_sql(
            f'select * from index_daily \
                where ts_code="{index_code}" and trade_date>="{start_date}" and trade_date<="{end_date}"',
            self.db_engine)
        return df

    # 指数日线行情
    def index_weekly(self, index_code, start_date, end_date):
        df = pd.read_sql(
            f'select * from index_weekly \
                where ts_code="{index_code}" and trade_date>="{start_date}" and trade_date<="{end_date}"',
            self.db_engine)
        return df

    # 返回指数包含的股票
    def index_weight(self, index_code, start_date, end_date):
        df = pd.read_sql(
            f'select * from index_weight \
                where index_code="{index_code}" and trade_date>="{start_date}" and trade_date<="{end_date}"',
            self.db_engine)
        return df['con_code'].unique().tolist()

    # 获得财务数据
    def fina_indicator(self, stock_code, start_date, end_date):
        stock_codes = db_utils.list_to_sql_format(stock_code)
        df = pd.read_sql(
            f'select * from fina_indicator where ts_code in ({stock_codes}) and ann_date>="{start_date}" and ann_date<="{end_date}"',
            self.db_engine)
        return df

    # 获得现金流量
    def income(self, stock_code, start_date, end_date):
        stock_codes = db_utils.list_to_sql_format(stock_code)
        df = pd.read_sql(
            f'select * from income where ts_code in ({stock_codes}) and ann_date>="{start_date}" and ann_date<="{end_date}"',
            self.db_engine)
        return df

    # 获得资产负债表
    def balance_sheet(self, stock_code, start_date, end_date):
        stock_codes = db_utils.list_to_sql_format(stock_code)
        df = pd.read_sql(
            f'select * from balancesheet \
                where ts_code in ({stock_codes}) and ann_date>="{start_date}" and ann_date<="{end_date}"',
            self.db_engine)
        return df

    # 获得现金流量表
    def cashflow(self, stock_code, start_date, end_date):
        stock_codes = db_utils.list_to_sql_format(stock_code)
        df = pd.read_sql(
            f'select * from cashflow \
                where ts_code in ({stock_codes}) and ann_date>="{start_date}" and ann_date<="{end_date}"',
            self.db_engine)
        return df

    def trade_cal(self, start_date, end_date, exchange='SSE'):
        df = pd.read_sql(
            f'select * from trade_cal where exchange="{exchange}" and cal_date>="{start_date}" and cal_date<="{end_date}" and is_open=1',
            self.db_engine)
        return df['cal_date']

    def stock_basic(self, ts_code=None):
        if ts_code is None or ts_code == "":
            return pd.read_sql(f'select * from stock_basic', self.db_engine)

        stock_codes = db_utils.list_to_sql_format(ts_code)
        df = pd.read_sql(f'select * from stock_basic where ts_code in ({stock_codes})', self.db_engine)
        return df

    def stock_holder_number(self, ts_code, start_date, end_date):
        stock_codes = db_utils.list_to_sql_format(ts_code)
        df = pd.read_sql(
            f'select * from stk_holdernumber where ts_code in ({stock_codes}) and ann_date>="{start_date}" and ann_date<="{end_date}"',
            self.db_engine)
        return df

    def index_classify(self, level='', src='SW2014'):
        df = pd.read_sql(f'select * from index_classify where src = \'{src}\'', self.db_engine)
        return df

    def get_factor(self, name, stock_codes, start_date, end_date):
        if not db_utils.is_table_exist(self.db_engine, f"factor_{name}"):
            raise ValueError(f"因子表factor_{name}在数据库中不存在")

        stock_codes = db_utils.list_to_sql_format(stock_codes)
        sql = f"""
            select * 
            from factor_{name} 
            where datetime>=\'{start_date}\' and 
                  datetime<=\'{end_date}\' and
                  code in ({stock_codes})
        """
        df = pd.read_sql(sql, self.db_engine)
        return df

    def limit_list(self):
        return pd.read_sql(f'select * from limit_list', self.db_engine)
