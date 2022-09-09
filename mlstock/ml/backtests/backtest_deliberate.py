import logging

from mlstock.const import TOP_30
from mlstock.data.datasource import DataSource
from mlstock.ml.backtests import predict, select_top_n, plot, timing
from mlstock.ml.backtests.broker import Broker
from mlstock.ml.backtests.metrics import metrics

logger = logging.getLogger(__name__)


def load_datas(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names):
    datasource = DataSource()

    df_data = predict(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names)
    df_limit = datasource.limit_list()
    df_index = datasource.index_weekly('000001.SH', start_date, end_date)
    ts_codes = df_data.ts_code.unique().tolist()
    df_daily = datasource.daily(ts_codes, start_date, end_date, adjust='')
    df_daily = df_daily.sort_values(['trade_date', 'ts_code'])
    df_calendar = datasource.trade_cal(start_date, end_date)
    df_baseline = df_data[['trade_date', 'next_pct_chg_baseline']].drop_duplicates()
    return df_data, df_daily, df_index, df_baseline, df_limit, df_calendar


def run_broker(df_data, df_daily, df_index, df_baseline, df_limit, df_calendar,
               start_date, end_date, factor_names,
               top_n, df_timing):
    df_selected_stocks = select_top_n(df_data, df_limit, top_n)

    broker = Broker(df_selected_stocks, df_daily, df_calendar, conservative=False, df_timing=df_timing)
    broker.execute()
    df_portfolio = broker.df_values
    df_portfolio.sort_values('trade_date')

    # 只筛出来周频的市值来
    df_portfolio = df_baseline.merge(df_portfolio, how='left', on='trade_date')

    # 拼接上指数
    df_index = df_index[['trade_date', 'close']]
    df_index = df_index.rename(columns={'close': 'index_close'})
    df_portfolio = df_portfolio.merge(df_index, how='left', on='trade_date')

    # 准备pct、next_pct_chg、和cumulative_xxxx
    df_portfolio = df_portfolio.sort_values('trade_date')
    df_portfolio['pct_chg'] = df_portfolio.total_value.pct_change()
    df_portfolio['next_pct_chg'] = df_portfolio.pct_chg.shift(-1)
    df_portfolio[['cumulative_pct_chg', 'cumulative_pct_chg_baseline']] = \
        df_portfolio[['next_pct_chg', 'next_pct_chg_baseline']].apply(lambda x: (x + 1).cumprod() - 1)

    df_portfolio = df_portfolio[~df_portfolio.cumulative_pct_chg.isna()]

    save_path = 'data/plot_{}_{}_top{}.jpg'.format(start_date, end_date, top_n)
    plot(df_portfolio, save_path)

    # 计算各项指标
    logger.info("佣金总额：%.2f", broker.total_commission)
    metrics(df_portfolio)


def main(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names):
    """
    先预测出所有的下周收益率、下周涨跌 => df_data，
    然后选出每周的top30 => df_selected_stocks，
    然后使用Broker，来遍历每天的交易，每周进行调仓，并，记录下每周的股票+现价合计价值 => df_portfolio
    最后计算出next_pct_chg、cumulative_pct_chg，并画出plot，计算metrics
    """
    df_data, df_daily, df_index, df_baseline, df_limit, df_calendar = \
        load_datas(data_path, start_date, end_date, model_pct_path, model_winloss_path, factor_names)

    df_timing = timing.ma(df_index)

    run_broker(df_data, df_daily, df_index, df_baseline, df_limit, df_calendar, start_date, end_date, factor_names,
               TOP_30, df_timing)
