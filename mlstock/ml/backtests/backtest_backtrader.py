import argparse
import logging
import os
import time

import backtrader as bt  # 引入backtrader框架
import backtrader.analyzers as bta  # 添加分析函数
import joblib
from backtrader.feeds import PandasData
from pandas import DataFrame
import numpy as np
from mlstock.const import TOP_30
from mlstock.data.datasource import DataSource
from mlstock.ml import load_and_filter_data
from mlstock.ml.backtests.ml_strategy import MachineLearningStrategy
from mlstock.ml.data import factor_conf
from mlstock.utils import utils, df_utils
from mlstock.utils.utils import AStockPlotScheme

logger = logging.getLogger(__name__)


def _select_top_n(df, df_limit):
    # 先把所有预测为跌的全部过滤掉
    original_size = len(df)
    df = df[df.winloss_pred == 1]
    logger.debug("根据涨跌模型结果，过滤数据 %d=>%d", original_size, len(df))

    df_limit = df_limit[['trade_date', 'ts_code', 'name']]
    df = df.merge(df_limit, on=['ts_date', 'trade_date'], how='left')
    original_size = len(df)
    df = df[~df.name.isna()]
    logger.debug("根据涨跌停信息，过滤数据 %d=>%d", original_size, len(df))

    # 先按照日期 + 下周预测收益，按照降序排
    df = df.sort_values(['trade_date', 'pct_pred'], ascending=False)

    # 用于保存选择后的股票，特别是他们的下期的实际收益率
    df_selected_stocks = DataFrame()

    # 按照日期分组，每组里面取前30，然后算收益率，作为组合资产的收益率
    # 注意！这里是下期收益"next_pct_chg"的均值，实际上是提前了一期（这个细节可以留意一下）
    df_groups = df.groupby('trade_date')
    for date, df_group in df_groups:
        # 根据 "预测收益率" 选出收益率top30
        df_top30 = df_group.iloc[0:TOP_30, :]

        # 根据 "实际收益率" 对这些选中股票求平均收益率（作为资产组合的收益率）
        next_pct_chg_mean = np.mean(df_top30.next_pct_chg.values)

        # 对基准的实际收益率也求一个平均（其实她们每个股票的这个值都是一样的，相加再去平均数，其实还是原来的数）
        next_pct_chg_baseline_mean = np.mean(df_top30.next_pct_chg_baseline.values)

        df_portfolio_pct = df_portfolio_pct.append([[date, next_pct_chg_mean, next_pct_chg_baseline_mean]])
        df_top30['trade_date'] = date
        df_selected_stocks = df_selected_stocks.append(df_top30)

    # 处理选中的股票的信息，保存下来，其实没啥用，就是存一下，方便细排查
    df_selected_stocks = df_selected_stocks[
        ['trade_date', 'ts_code', 'target', 'pct_pred', 'next_pct_chg', 'next_pct_chg_baseline']]
    df_selected_stocks.columns = [
        'trade_date', 'ts_code', 'target', 'pct_pred', 'next_pct_chg', 'next_pct_chg_baseline']
    df_selected_stocks.to_csv("data/top30.csv", header=0)
    return df_selected_stocks


def load_data_to_cerebro(cerebro, start_date, end_date, df):
    df = df.rename(columns={'vol': 'volume', 'ts_code': 'name', 'trade_date': 'datetime'})  # 列名准从backtrader的命名规范
    df['openinterest'] = 0  # backtrader需要这列，所以给他补上
    df['datetime'] = df_utils.to_datetime(df['datetime'], date_format="%Y%m%d")
    df = df.set_index('datetime')
    df = df.sort_index()
    d_start_date = utils.str2date(start_date)  # 开始日期
    d_end_date = utils.str2date(end_date)  # 结束日期
    data = PandasData(dataname=df, fromdate=d_start_date, todate=d_end_date, plot=True)
    cerebro.adddata(data)
    logger.debug("初始化股票%s~%s数据到脑波cerebro：%d 条", start_date, end_date, len(df))


def main(data_path, start_date, end_date, model_pct_path, model_winloss_path,factor_names):
    """
    datetime    open    high    low     close   volume  openi..
    2016-06-24	0.16	0.002	0.085	0.078	0.173	0.214
    2016-06-30	0.03	0.012	0.037	0.027	0.010	0.077
    """
    cerebro = bt.Cerebro()  # 初始化cerebro
    datasource = DataSource()

    # 从csv因子数据文件中加载数据
    df_data = load_and_filter_data(data_path, start_date, end_date)
    df_daily = datasource.daily(df_data.ts_codes)
    df_limit = datasource.limit_list()
    df_index = datasource.index_weight('00001.SH', start_date, end_date)

    # 加载模型；如果参数未提供，为None
    # 查看数据文件和模型文件路径是否正确
    if model_pct_path: utils.check_file_path(model_pct_path)
    if model_winloss_path: utils.check_file_path(model_winloss_path)
    model_pct = joblib.load(model_pct_path) if model_pct_path else None
    model_winloss = joblib.load(model_winloss_path) if model_winloss_path else None
    if model_pct:
        start_time = time.time()
        X = df_data[factor_names]
        df_data['pct_pred'] = model_pct.predict(X)
        utils.time_elapse(start_time, f"预测下期收益: {len(df_data)}行 ")
    if model_winloss:
        start_time = time.time()
        X = df_data[factor_names]
        df_data['winloss_pred'] = model_winloss.predict(X)
        utils.time_elapse(start_time, f"预测下期涨跌: {len(df_data)}行 ")
    df_selected_stocks = _select_top_n(df_data, df_limit)

    # 加载股票数据到脑波
    load_data_to_cerebro(cerebro, start_date, end_date, df_daily)

    start_cash = 500000.0
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.002)
    cerebro.addstrategy(MachineLearningStrategy, df_data=df_selected_stocks)

    # 添加分析对象
    cerebro.addanalyzer(bta.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)  # 夏普指数
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DW')  # 回撤分析
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.Calmar, _name='calmar')  # 卡玛比率 - Calmar：超额收益➗最大回撤
    cerebro.addanalyzer(bt.analyzers.PeriodStats, _name='period_stats')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')  # 加入PyFolio分析者,这个是为了做quantstats分析用
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')

    # 打印
    logger.debug('回测期间：%r ~ %r , 初始资金: %r', start_date, end_date, start_cash)
    # 运行回测
    results = cerebro.run(optreturn=True)

    if not os.path.exists(f"debug/"): os.makedirs(f"debug/")
    file_name = f"debug/{utils.nowtime()}_{start_date}_{end_date}.html"

    b = Bokeh(filename=file_name, style='bar', plot_mode='single', output_mode='save', scheme=AStockPlotScheme())
    cerebro.plot(b, style='candlestick', iplot=False)


"""
# 测试用
python -m mfm_learner.example.factor_backtester \
    --factor momentum_3m \
    --start 20180101 \
    --end 20191230 \
    --num 50 \
    --period 20 \
    --index 000905.SH \
    --risk
    
python -m mfm_learner.example.factor_backtester \
    --factor clv \
    --start 20180101 \
    --end 20190101 \
    --index 000905.SH \
    --num 20 \
    --period 20 \
    --risk  \
    --percent 10%   

python -m mfm_learner.example.factor_backtester \
    --factor synthesis:clv_peg_mv \
    --start 20180101 \
    --end 20190101 \
    --num 20 \
    --period 20 \
    --index 000905.SH

python -m mfm_learner.example.factor_backtester \
    --factor clv,peg,mv,roe_ttm,roe_ \
    --start 20180101 \
    --end 20190101 \
    --num 20 \
    --period 20 \
    --index 000905.SH
"""
if __name__ == '__main__':
    utils.init_logger()

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--factor', type=str, help="单个因子名、多个（逗号分割）、所有（all）")
    parser.add_argument('-s', '--start', type=str, help="开始日期")
    parser.add_argument('-e', '--end', type=str, help="结束日期")
    parser.add_argument('-i', '--index', type=str, help="股票池code")
    parser.add_argument('-p', '--period', type=int, help="调仓周期，多个的话，用逗号分隔")
    parser.add_argument('-an', '--atr_n', type=int, default=3, help="ATR风控倍数")
    parser.add_argument('-ap', '--atr_p', type=int, default=15, help="ATR周期")
    parser.add_argument('-n', '--num', type=int, help="股票数量")
    parser.add_argument('-r', '--risk', action='store_true', help="是否风控")
    args = parser.parse_args()

    main(args.start,
         args.end,
         args.index,
         args.period,
         args.num,
         args.factor,
         args.risk,
         args.atr_p,
         args.atr_n)
    logger.debug("共耗时: %.0f 秒", time.time() - start_time)
