"""
只保留非ST，主板，中小板，剔除（ST、创业板、科创板、北交所、B股）
```
- 证券交易所的地点：沪市是上海；深市是深圳;北交所是北京；
- 板块不同：沪市只有主板与B股；深市有主板、中小板、创业板和B股。
- 股票代码不同：沪市主板是60开头，B股是900开头；深市主板是000开头，中小板是002开头、创业板是300开头、B股是200开头
- 新股上市首日不设涨跌幅限制(沪深两市股票上市首日不设涨跌幅,创业板、科创板新股上市前五个交易日不设涨跌幅)。
- 科创板股票竞价交易实行价格涨跌幅限制，涨跌幅比例为20%。 首次公开发行上市的科创板股票，上市后的前5个交易日不设价格涨跌幅限制
```
"""
from mlstock import const
from mlstock.const import STOCK_IPO_YEARS
from mlstock.data.datasource import DataSource
from mlstock.utils import utils
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def filter_BJ_Startup_B(df_stocks):
    df = df_stocks[(df_stocks.market == "主板") | (df_stocks.market == "中小板")]
    logger.debug("从[%d]只过滤掉非主板、非中小板股票后，剩余[%d]只股票", len(df_stocks), len(df))
    return df


def filter_stocks(least_year=STOCK_IPO_YEARS):
    """
    用于全市场过滤股票，默认的标准是：至少上市1年的
    从stock_basic + daily_basic，合并，原因是daily_basic有市值信息
    max_circ_mv: 最大流动市值，单位是亿
    """
    datasource = DataSource()
    df_stocks = datasource.stock_basic()

    total_amount = len(df_stocks)
    logger.debug("加载[%d]只股票的基础信息(basic)", total_amount)

    df_stocks = filter_unlist(df_stocks, total_amount)

    df_stocks = filter_by_years(df_stocks, end_date=utils.today(), least_year=least_year)

    df_stocks = filter_ST(df_stocks)

    df_stocks = filter_BJ_Startup_B(df_stocks)

    return df_stocks['ts_code'].tolist()


def filter_ST(df_stocks):
    # 剔除ST股票
    total_amount = len(df_stocks)
    df_stocks = df_stocks[~df_stocks['name'].str.contains("ST")]
    logger.debug("过滤掉ST的[%d]只股票后，剩余[%d]只股票", total_amount - len(df_stocks), len(df_stocks))
    return df_stocks


def filter_by_years(df_stocks, end_date, least_year):
    df_stocks['list_date'] = pd.to_datetime(df_stocks['list_date'], format='%Y%m%d')
    df_stocks['period'] = utils.str2date(end_date) - df_stocks['list_date']
    df_stocks_more_years = df_stocks[df_stocks['period'] > pd.Timedelta(days=365 * least_year)]
    logger.debug("过滤掉上市不到[%d]年[%d]只的股票，剩余[%d]只股票", least_year, len(df_stocks) - len(df_stocks_more_years),
                 len(df_stocks_more_years))
    return df_stocks_more_years


def filter_unlist(df_stocks, total_amount):
    print(df_stocks)
    df_stocks = df_stocks[df_stocks['list_status'] == 'L']
    logger.debug("过滤掉[%d]只不在市股票，剩余[%d]只股票", total_amount - len(df_stocks), len(df_stocks))
    return df_stocks


# python -m mlstock.data.data_filter
if __name__ == '__main__':
    utils.init_logger(simple=True)
    print(filter_stocks())
