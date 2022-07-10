import argparse
import logging

import pandas as pd
from tabulate import tabulate

from utils import utils, db_utils

logger = logging.getLogger(__name__)

TOP = 100


def _table(df):
    return tabulate(df, headers='keys', tablefmt='psql')


def main(strategy_name, start_date, end_date):
    db_engine = utils.connect_db()
    df = load_strategy(db_engine, strategy_name, start_date, end_date)
    if df is None or len(df) == 0:
        logger.debug("无法检索到任何统计数据")
        return

    # df_start = df.loc[0,'开始日期']
    # df_end = df.loc[0, '结束日期']
    # print(df_start,df_end)
    pd.options.display.float_format = '{:,.1f}'.format
    df_positive = df[df['盈亏比例'] > 0]

    logger.debug("盈钱情况：")
    logger.debug(_table(df_positive.describe()))
    df_negative = df[df['盈亏比例'] < 0]

    logger.debug("亏钱情况：")
    logger.debug(_table(df_negative.describe()))

    logger.debug(f"盈利Top{TOP}：")
    logger.debug(_table(df_positive.sort_values('盈亏比例', ascending=False).head(TOP)))

    logger.debug(f"亏损Top{TOP}：")

    df_detail = load_detail(db_engine, strategy_name)

    df_negative = df_detail[['股票代码', '买入日期', '卖出日期']].merge(df_negative, on='股票代码')

    logger.debug(_table(df_negative.sort_values('盈亏比例').head(TOP)))

    # 年分布
    detail(df_detail, strategy_name)

    win_rate = 100 * len(df_positive) / (len(df_positive) + len(df_negative))
    logger.debug("盈利股票数量：%d 只", len(df_positive))
    logger.debug("亏损股票数量：%d 只", len(df_negative))
    logger.debug("胜率（股票）：%.2f%%" % win_rate)


def load_strategy(db_engine,strategy_name,start_date=None, end_date=None):

    if not db_utils.is_table_exist(db_engine,"cta_stats"): return None

    if start_date is None and end_date is None:
        sql = f'''
            select 股票代码,开始日期,结束日期,盈亏比例,总交易数,交易总日,交易均日,交易最长,交易最短 
            from cta_stats
            where 策略名称='{strategy_name}'
        '''
    else:
        sql = f'''
        select 股票代码,开始日期,结束日期,盈亏比例,总交易数,交易总日,交易均日,交易最长,交易最短 
        from cta_stats 
        where 
            策略名称='{strategy_name}' and
            开始日期='{start_date}' and
            结束日期='{end_date}'
        '''
    # logger.debug(sql)
    logger.debug("查询：%s~%s 的 策略[%s]数据", start_date, end_date, strategy_name)
    df = pd.read_sql(sql, db_engine)
    return df


def load_detail(db_engine, strategy_name):
    sql = f'''
        select * 
        from cta_stats_detail 
        where 策略名称="{strategy_name}" 
        '''
    df = pd.read_sql(sql, db_engine)
    return df


def detail(df, strategy_name):
    logger.debug("查询：策略[%s]数据", strategy_name)
    df_positive = df[df['收益率'] > 0]
    df_negative = df[df['收益率'] < 0]
    logger.debug("一共的交易机会有：%d 次", len(df))
    logger.debug("赚钱的交易机会有：%d 次", len(df_positive))
    logger.debug("亏钱的交易机会有：%d 次", len(df_negative))
    logger.debug("胜率（交易机会）：%.2f%%", 100 * len(df_positive) / (len(df_positive) + len(df_negative) + 0.00001))

    # 按照年统计
    df['买入年份'] = df['买入日期'].str[:4]
    logger.debug("每年买入的次数统计：")
    df_count = df[['买入年份', '股票代码']].groupby('买入年份').count()
    df_count.columns = ['数量']
    logger.debug(df_count)
    logger.debug("一共测试的年分数：%.1f年", len(df_count))
    logger.debug("一共的买入的机会：%.1f次", df_count.sum().item())
    logger.debug("每年平均买入机会：%.1f次", df_count.mean().item())


# python -m utils.stat -t cao.v2 -s 20080101 -e 20220501
# python -m utils.stat -t cao.v6:gap=0.1,macd=0.2,atr=0.8  -s 20080101 -e 20220501
if __name__ == '__main__':
    utils.init_logger(simple=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--strategy', type=str, help="策略名称")
    parser.add_argument('-s', '--start_date', type=str, help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, help="结束日期")
    args = parser.parse_args()
    main(args.strategy, args.start_date, args.end_date)

"""
策略名称：cao.v2
股票代码：200021.SZ
买入日期：20160405
卖出日期：20160816
策略收益：-19.92
 [20160405]~[20160816] 持仓8交易日，策略收益：股票[002199.SZ], 收益率[-19.92%]
"""
