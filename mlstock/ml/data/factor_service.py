import argparse
import json
import logging
import os
import time

import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlstock.const import CODE_DATE, BASELINE_INDEX_CODE, TARGET
from mlstock.data import data_filter, data_loader
from mlstock.data.datasource import DataSource
from mlstock.data.stock_info import StocksInfo
from mlstock.ml.data import factor_conf
from mlstock.ml.data.factor_conf import FACTORS
from mlstock.utils import utils
from mlstock.utils.industry_neutral import IndustryMarketNeutral
from mlstock.utils.utils import time_elapse

logger = logging.getLogger(__name__)


def calculate(start_date, end_date, num, is_industry_neutral):
    """
    从头开始计算因子
    :param start_date:
    :param end_date:
    :param num:
    :return:
    """

    start_time = time.time()
    data_source = DataSource()

    # 加载股票数据
    stock_data, ts_codes = load_stock_data(data_source, start_date, end_date, num)

    # 加载（计算）因子
    df_weekly, factor_names = calculate_factors(data_source, stock_data, StocksInfo(ts_codes, start_date, end_date))

    # 显存一份最原始的数据
    time_elapse(start_time, "⭐️ 全部因子加载完成")

    # 加载基准（指数）数据
    df_weekly = prepare_target(df_weekly, start_date, end_date, data_source)

    # 清晰因子数据
    df_weekly = clean_factors(df_weekly, factor_names, start_date, end_date, is_industry_neutral)

    # 保存原始数据和处理后的数据
    # save_csv("raw", df_weekly, start_date, end_date)
    save_csv("processed" + ("_industry_neutral" if is_industry_neutral else ""), df_weekly, start_date, end_date)

    return df_weekly, factor_names


def load_stock_data(data_source, start_date, end_date, num):
    """
    筛选出合适的股票，并，加载数据

    :param data_source:
    :param start_date:
    :param end_date:
    :param num:
    :return:
    """

    # 过滤非主板、非中小板股票、且上市在1年以上的非ST股票
    df_stock_basic = data_filter.filter_stocks()
    df_stock_basic = df_stock_basic.iloc[:num]
    df_stock_basic = process_industry(df_stock_basic)  # 把industry列换成ID

    ts_codes = df_stock_basic.ts_code

    # 临时保存一下，用于本地下载数据提供列表（调试用）
    # df_stock_basic.ts_code.to_csv("data/stocks.txt", index=False)

    # 加载周频数据
    stock_data = data_loader.load(data_source, ts_codes, start_date, end_date)

    # 把基础信息merge到周频数据中
    df_weekly = stock_data.df_weekly.merge(df_stock_basic, on='ts_code', how='left')

    # 某只股票上市12周内的数据扔掉，不需要
    old_length = len(df_weekly)
    a = pd.to_datetime(df_weekly.trade_date, format='%Y%m%d')
    b = pd.to_datetime(df_weekly.list_date, format='%Y%m%d')
    df_weekly = df_weekly[a - b > pd.Timedelta(12, unit='w')]
    logger.info("剔除掉上市12周内的数据：%d=>%d", old_length, len(df_weekly))

    stock_data.df_weekly = df_weekly
    return stock_data, ts_codes


def calculate_factors(data_source, stock_data, stocks_info):
    """
    计算每一个因子，因子列表来自于factor_conf.py

    :param data_source:
    :param stock_data:
    :param stocks_info:
    :return:
    """

    logger.info("加载和清洗数据")

    factor_names = []
    df_weekly = stock_data.df_weekly

    # 获取每一个因子（特征），并且，并入到股票数据中
    for factor_class in FACTORS:
        factor = factor_class(data_source, stocks_info)
        df_factor = factor.calculate(stock_data)
        df_weekly = factor.merge(df_weekly, df_factor)
        factor_names += factor.name if type(factor.name) == list else [factor.name]
        logger.info("获取因子%r %d 行数据", factor.name, len(df_factor))

    logger.info("因子加载完成，合计 %d 行数据，%d个因子:\n%r", len(df_weekly), len(factor_names), factor_names)
    return df_weekly, factor_names


def load_from_file(factors_file_path):
    """
    从文件中，直接加载因子数据

    :param factors_file_path:
    :return:
    """
    if not os.path.exists(factors_file_path):
        raise ValueError(f"因子数据文件不存在：{factors_file_path}")
    df_features = pd.read_csv(factors_file_path, header=0)
    df_features['trade_date'] = df_features['trade_date'].astype(str)
    df_features['target'] = df_features['target'].astype(int)
    return df_features


def extract_features(df):
    return df[factor_conf.get_factor_names()]


def extract_train_data(df):
    """
    仅保留训练用的列
    :param factors_file_path:
    :return:
    """
    return df[CODE_DATE + factor_conf.get_factor_names() + TARGET]


def prepare_target(df_weekly, start_date, end_date, datasource):
    """
    计算基准的收益率等各种预测用的收益率
    :param df_weekly:
    :param datasource:
    :return:
    """

    # 合并沪深300的周收益率，为何用它呢，是为了计算超额收益(r_i = pct_chg - next_pct_chg_baseline)
    df_baseline = datasource.index_weekly(BASELINE_INDEX_CODE, start_date, end_date)
    df_baseline = df_baseline[['trade_date', 'pct_chg']]
    df_baseline = df_baseline.rename(columns={'pct_chg': 'pct_chg_baseline'})
    logger.info("下载基准[%s] %s~%s 数据 %d 条", BASELINE_INDEX_CODE, start_date, end_date, len(df_baseline))

    df_weekly = df_weekly.merge(df_baseline, on=['trade_date'], how='left')
    logger.info("合并基准[%s] %d=>%d", BASELINE_INDEX_CODE, len(df_weekly), len(df_weekly))

    # 计算出和基准的超额收益率，并且基于它，设置预测标签'target'（预测下一期，所以做shift）
    df_weekly['rm_rf'] = df_weekly.pct_chg - df_weekly.pct_chg_baseline

    # target即预测目标，是"下一期"的超额收益，训练主要靠这个，他就是训练的y
    df_weekly['target'] = df_weekly.groupby('ts_code').rm_rf.shift(-1)

    # 下一期的收益率，这个是为了将来做回测评价用，注意，是下一期，所以做了shift(-1)
    df_weekly['next_pct_chg'] = df_weekly.groupby('ts_code').pct_chg.shift(-1)
    df_weekly['next_pct_chg_baseline'] = df_weekly.groupby('ts_code').pct_chg_baseline.shift(-1)

    return df_weekly


def _scaller(x, df_median, df_scope):
    """
    - 则将序列𝐷𝑖中所有大于𝐷𝑀 + 5𝐷𝑀1的数重设为𝐷𝑀 + 5𝐷𝑀1
    - 将序列𝐷𝑖中所有小于𝐷𝑀 − 5𝐷𝑀1的数重设为𝐷𝑀 − 5𝐷𝑀1
    :param x: 就是某一列，比如beta
        Name: beta, Length: 585, dtype: float64
        180          NaN
        181          NaN
                  ...
        1196    163121.0
    :param df_median:
        (Pdb) df_median
        return_1w                 -0.002050
        return_3w                 -0.007407
        .....                     ......
        alpha                     0.000161
        beta                      0.276572
        stake_holder              163121.000000
        Length: 73, dtype: float64
    :param df_scope:
        (Pdb) df_scope
        return_1w                 0.029447
        .....                     ......
        stake_holder              82657.000000
        Length: 73, dtype: float64
    :return:
    """
    _max = df_median[x.name] + 5 * df_scope[x.name]
    _min = df_median[x.name] - 5 * df_scope[x.name]
    x = x.apply(lambda v: _min if v < _min else v)
    x = x.apply(lambda v: _max if v > _max else v)
    return x


def clean_factors(df_weekly, factor_names, start_date, end_date, is_industry_market_neutral):
    """
    对因子数据做进一步的清洗，这步很重要，也很慢
    :param df_features:
    :param factor_names:
    :param start_date: 因为前面的日期中，为了防止MACD之类的技术指标出现NAN预加载了数据，所以要过滤掉这些start_date之前的数据
    :return:
    """

    start_time = time.time()

    """
    因为前面的日期中，为了防止MACD之类的技术指标出现NAN预加载了数据，所以要过滤掉这些start_date之前的数据
    """
    original_length = len(df_weekly)
    df_weekly = df_weekly[df_weekly.trade_date >= start_date]
    logger.info("过滤掉[%s]之前的数据（为防止技术指标nan）后：%d => %d 行", start_date, original_length, len(df_weekly))

    logger.info("(调试)特征处理之前的数据情况：\n%r", df_weekly[CODE_DATE + factor_names].describe())

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("(调试)特征处理之前NA统计：数据特征中的NAN数：\n%r", df_weekly[factor_names].isna().sum().sort_values())

    """
    如果target缺失比较多，就删除掉这些股票
    """
    original_length = len(df_weekly)
    df_weekly = df_weekly[~df_weekly.target.isna()]
    logger.info("过滤掉target为nan的行后：%d => %d 行，剔除占比%.1f%%",
                original_length,
                len(df_weekly),
                (original_length - len(df_weekly)) * 100 / original_length)

    """
    去除那些因子值中超过20%缺失的股票（看所有因子中确实最大的那个，百分比超过20%，这只股票整个剔除掉）
    """
    # 计算每只股票的每个特征的缺失百分比
    # 仅用[ 特征s, 股票,日期 ] 这些列作为统计手段
    df_na_miss_percent_by_code = df_weekly[CODE_DATE + factor_names].groupby(by='ts_code').apply(
        lambda df: (df.shape[0] - df.count()) / df.shape[0])

    # 找出最大的那个特征的缺失比，如果其>80%，就剔除这只股票
    df_na_miss_codes = df_na_miss_percent_by_code[df_na_miss_percent_by_code.max(axis=1) > 0.8]['ts_code']
    # 把这些行找出来，打印到日志中，方便后期调试
    df_missed_info = df_na_miss_percent_by_code[
        df_na_miss_percent_by_code.apply(lambda x: x.name in df_na_miss_codes, axis=1)]
    # 0缺失的列，需要扣掉，只保留确实列打印出来调试
    need_drop_columns = df_missed_info.sum()[df_missed_info.sum() == 0].index
    # 仅保留确实存在确实的列，打印出来调试
    df_missed_info = df_missed_info.drop(need_drop_columns, axis=1)
    logger.info("(调试)以下股票的某些特征的'缺失(NA)率'，超过80%%，%d 只(需要被删掉的股票)：\n%r", len(df_missed_info), df_missed_info)
    # 剔除这些问题股票
    origin_stock_size = len(df_weekly.ts_code.unique())
    origin_data_size = df_weekly.shape[0]
    df_weekly = df_weekly[df_weekly.ts_code.apply(lambda x: x not in df_na_miss_codes)]
    logger.info("从%d只股票中剔除了%d只，占比%.1f%%；剔除相关数据%d=>%d行，剔除占比%.2f%%",
                origin_stock_size,
                len(df_na_miss_codes),
                len(df_na_miss_codes) * 100 / origin_stock_size,
                origin_data_size,
                len(df_weekly),
                (origin_data_size - len(df_weekly)) * 100 / origin_data_size)

    """
    去除极值+标准化
    每一列，都去极值（TODO：是不是按照各股自己的值来做是不是更好？现在是所有的股票）
    中位数去极值:
    - 设第 T 期某因子在所有个股上的暴露度序列为𝐷𝑖
    - 𝐷𝑀为该序列中位数
    - 𝐷𝑀1为序列|𝐷𝑖 − 𝐷𝑀|的中位数
    - 则将序列𝐷𝑖中所有大于𝐷𝑀 + 5𝐷𝑀1的数重设为𝐷𝑀 + 5𝐷𝑀1
    - 将序列𝐷𝑖中所有小于𝐷𝑀 − 5𝐷𝑀1的数重设为𝐷𝑀 − 5𝐷𝑀1
    """
    # 每列都求中位数，和中位数之差的绝对值的中位数
    df_features_only = df_weekly[factor_names]
    # 找到每一个特征的中位值
    df_median = df_features_only.median()
    # 每个值，都和中位数相减后，取绝对值，然后在找到绝对值们的中位数，这个就是要限定的范围值
    df_scope = df_features_only.apply(lambda x: x - df_median[x.name]).abs().median()
    df_features_only = df_features_only.apply(lambda x: _scaller(x, df_median, df_scope))

    # 标准化：
    # 将中性化处理后的因子暴露度序列减去其现在的均值、除以其标准差，得到一个新的近似服从N(0,1)分布的序列。
    scaler = StandardScaler()
    scaler.fit(df_features_only)
    df_weekly[factor_names] = scaler.transform(df_features_only)
    logger.info("对%d个特征进行了标准化(中位数去极值)处理：%d 行", len(factor_names), len(df_weekly))

    # 去除所有的NAN数据(with用来显示所有航)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("NA统计：数据特征中的NAN数：\n%r", df_weekly[factor_names].isna().sum().sort_values())
    df_weekly = filter_invalid_data(df_weekly, factor_names)

    original_length = len(df_weekly)
    df_weekly.dropna(subset=factor_names + ['target'], inplace=True)
    logger.info("去除NAN后，数据剩余行数：%d=>%d 行，剔除了%.1f%%",
                original_length,
                len(df_weekly),
                (original_length - len(df_weekly)) * 100 / original_length)

    """
    去重
    """
    original_length = len(df_weekly)
    df_weekly = df_weekly[~df_weekly[CODE_DATE].duplicated()].reset_index(drop=True)
    logger.info("去除重复行(ts_code+trade_date)后，数据 %d => %d 行，剔除了%.1f%%",
                original_length,
                len(df_weekly),
                (original_length - len(df_weekly)) * 100 / original_length)

    # 行业中性化处理
    if is_industry_market_neutral:
        start_time1 = time.time()
        industry_market_neutral = IndustryMarketNeutral(factor_names,
                                                        market_value_name='total_market_value_log',
                                                        industry_name='industry')
        industry_market_neutral.fit(df_weekly)
        df_weekly = industry_market_neutral.transform(df_weekly)
        time_elapse(start_time1, "行业中性化处理")

    # 保存最后的训练数据：ts_code、trade_date、factors、target
    df_data = df_weekly[CODE_DATE + factor_names + ['target']]
    logger.info("特征处理之后的数据情况：\n%r", df_data.describe())

    time_elapse(start_time, "⭐️ 全部因子预处理完成")
    return df_weekly


def save_csv(name, df, start_date, end_date):
    csv_file_name = "data/{}_{}_{}_{}.csv".format(name, start_date, end_date, utils.now())
    df.to_csv(csv_file_name, header=True, index=False)  # 保留列名
    logger.info("保存 %d 行数据到文件：%s", len(df), csv_file_name)


def filter_invalid_data(df, factor_names):
    for factor_name in factor_names:
        original_size = len(df)
        # 去掉那些这个特征全是nan的股票
        valid_ts_codes = df.groupby('ts_code')[factor_name].count()[lambda x: x > 0].index
        df = df[df['ts_code'].isin(valid_ts_codes)]
        if len(df) != original_size:
            logger.info("去除特征[%s]全部为Nan的股票数据后，行数变化：%d => %d",
                        factor_name, original_size, len(df))
    return df


def process_industry(df_basic):
    name_id_mapping_file = 'data/industry.json'

    # 先存一下中文名，后面industry列会被数字替换
    df_basic['industry_cn'] = df_basic['industry']
    # 行业的缺失值使用其他填充
    df_miss_industry = df_basic[df_basic.industry.isna()]
    logger.warning("以下[%d]只股票，占比%.1f%%, 缺少行业信息：\n%r",
                   len(df_miss_industry),
                   len(df_miss_industry) * 100 / len(df_basic),
                   df_miss_industry)
    df_basic.industry = df_basic.industry.fillna('其他')

    # 加载 或 映射，行业的名字=>ID
    if os.path.exists(name_id_mapping_file):
        logger.info("行业名称/ID映射文件[%s]存在，使用它", name_id_mapping_file)
        with open(name_id_mapping_file, 'r') as f:
            industry_to_number = json.load(f)
        # 转成整形
        for k, v in industry_to_number.items():
            industry_to_number[k] = int(v)
    else:
        # 排个序，防止序号将来再运行乱掉
        industry_names = df_basic.industry.sort_values().unique()

        # 名字=>ID
        industry_to_number = {}
        for i, v in enumerate(industry_names):
            industry_to_number[v] = i + 1
        # 保存下来映射
        with open(name_id_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(industry_to_number, f, sort_keys=True, indent=4)
            logger.info("行业编码信息保存到：%s", name_id_mapping_file)

    # 转换数据中的行业：名称=>ID
    df_basic.industry = df_basic.industry.map(industry_to_number)
    logger.debug("行业ID映射：%r", industry_to_number)

    # 返回处理后的数据
    return df_basic


"""
python -m mlstock.ml.data.factor_service -n 50 -d -s 20080101 -e 20220901
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 数据相关的
    parser.add_argument('-s', '--start_date', type=str, default="20090101", help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, default="20220801", help="结束日期")
    parser.add_argument('-n', '--num', type=int, default=100000, help="股票数量，调试用")
    parser.add_argument('-in', '--industry_neutral', action='store_true', default=False, help="是否做行业中性处理")

    # 全局的
    parser.add_argument('-d', '--debug', action='store_true', default=True, help="是否调试")

    args = parser.parse_args()

    if args.debug:
        print("【调试模式】")
        utils.init_logger(file=True, log_level=logging.DEBUG)
    else:
        utils.init_logger(file=True, log_level=logging.INFO)

    calculate(args.start_date, args.end_date, args.num, args.industry_neutral)
