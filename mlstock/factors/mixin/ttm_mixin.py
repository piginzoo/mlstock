import logging
import numpy as np

from mlstock.utils import utils
from mlstock.utils.utils import logging_time

logger = logging.getLogger(__name__)


class TTMMixin:
    """
    处理TTM：以当天为基准，向前滚动12个月的数据，
    用于处理类ROE_TTM数据，当然不限于ROE，只要是同样逻辑的都支持。

    @:param finance_date  - 真正的财报定义的日期，如3.30、6.30、9.30、12.31

    ts_code    ann_date  end_date      roe
    600000.SH  20201031  20200930   7.9413
    600000.SH  20200829  20200630   5.1763
    600000.SH  20200425  20200331   3.0746
    600000.SH  20200425  20191231  11.4901
    600000.SH  20191030  20190930   9.5587 <----- 2019.8.1日可回溯到的日期
    600000.SH  20190824  20190630   6.6587
    600000.SH  20190430  20190331   3.4284
    600000.SH  20190326  20181231  12.4674

    处理方法：
    比如我要填充每一天的ROE_TTM，就要根据当前的日期，回溯到一个可用的ann_date（发布日期），
    然后以这个日期，作为可用数据，计算ROE_TTM。
    比如当前日是2019.8.1日，回溯到2019.10.30(ann_date)日发布的3季报（end_date=20190930, 0930结尾为3季报），
    然后，我们的计算方法就是，用3季报，加上去年的年报，减去去年的3季报。
    -----
    所以，我们抽象一下，所有的规则如下：
    - 如果是回溯到年报，直接用年报作为TTM
    - 如果回溯到1季报、半年报、3季报，就用其 + 去年的年报 - 去年起对应的xxx报的数据，这样粗暴的公式，是为了简单

    2022.7.22
    之前的处理是用具体的一天，去反向查找最近的发布日期(ann_date)，
    现在不能这样做了，现在是先按照ann_date去计算每个ann_date的ttm，
    然后再去反向填充每个周频交易日（一般是周五）
    思考：看之前是用上证交易日，得到日频，计算每日这只股票对应的TTM，这个计算量有点大，不这么做了。
    还是反向填充ann_date的ttm，这个计算量小很多了。
    ---
    这样实现的话，有个细节，ann_date并不是很规整的，不是3.31,6.30,9.30,12.31这些规整日，
    她可能任何一天发布，所以，先要把她们规整到这些规整日去，
    举个例子：
        ts_code    ann_date   f_ann_date  end_date report_type comp_type  basic_eps  diluted_eps
        600230.SH  20150815   20150815    20150630           1         1    -0.7972      -0.7972
    这个是2015.8.15号发布的2015年的半年报，
    现在我们要计算这条的TTM，也就是20150815发布的信息的TTM。
    那么计算公式是： 2015半年报 + 2014年报 - 2014半年报
    所以，我们要找到"2014的年报"，和，"2014的半年报"，
    这个需要去这只股票的财务数据中去检索，然后按照上述公式计算出来。
    如果找不到这2个数据（"2014的年报"，和，"2014的半年报"）中的任何一个，
    就只能用当前的半年报来近似了。

    ========
    注： TTM后，还需要填充处理，参见<FillMixin>

    ========
    关于财报：(财报总是累计值，如三季报是1-9月累计值）
    季报是指每三个月结束后的经营情况报表，三季报是指上市公司1月到9月的经营情况报表，
    即三季报是前三季度的经营情况总和的报表，而不单独是第三季度的经营情况报表。
    年报是指当年结束后全年的经营情况报表；中报是指半年结束后的经营情况报表；
    上市公司必须披露定期报告，它包括年度报告、中期报告、第一季报、第三季报。
    根据中国证监会《上市公司信息披露管理办法》的规定，上市公司年报的披露时间为每个会计年度结束之日起4个月内，即一至四月份，
    中期报告由上市公司在半年度结束后两个月内完成，即七、八月份，
    季报由上市公司在会计年度前三个月、九个月结束后的三十日内编制完成，即第一季报在四月份，第三季报在十月份
    """

    @logging_time
    def ttm(self, df, finance_column_names, publish_date_column_name='ann_date', finance_date_column_name='end_date'):
        """
        :param df: 包含了 ts_code, ann_date，<各种需要TTM处理的财务指标列> 的 dataframe
        :return: 财务指标列被替换成了TTM值
        """
        # df.fillna(np.NAN,inplace=True)

        # 删除重复值
        row_num = len(df)
        df = df[~df[['ts_code', publish_date_column_name]].duplicated(keep='last')]
        if len(df) < row_num:
            logger.warning("删除重复行数：%d", row_num - len(df))
            row_num = len(df)

        # 剔除Nan
        # df.dropna(inplace=True)
        if len(df) < row_num:
            if len(df) < row_num:
                logger.warning("删除包含NAN的行数：%d", row_num - len(df))

        # 对时间，升序排列
        df.sort_values(publish_date_column_name, inplace=True)

        return df.groupby('ts_code').apply(self.handle_one_stock_ttm,
                                           finance_column_names=finance_column_names,
                                           finance_date_column_name=finance_date_column_name)

    def handle_one_stock_ttm(self, df, finance_column_names,finance_date_column_name):
        """
        这个处理和致敬大神的不一样，它是假设所有的财报都有，这样错位一下(shift)就可以算出ttm了，
        举个例子：
                             (20213031)  (20201231)  (20200630)  (20200331)
        ts_code    ann_date  basic_eps_1 basic_eps_2 basic_eps_3 basic_eps_4
        600000.SH, 20210630, 1234,       4567,       8910,       1234
        basic_eps_(1-4)就是它上1-4期的数据，然后他就可以按照这4个生成列，计算了：
        ```
        def lastyear(onerow):
            end_date = str(onerow['end_date'])[-4:]
            if end_date=='0331':
                return onerow[bs_f[i]] + onerow[bs_f[i]+'{}'.format(1)] - onerow[bs_f[i]+'{}'.format(4)]
            ......
        ```
        但是，这里有个问题，如果某一期，这个公司没有发布年报、季报啥的，或者数据缺失了，
        她就会完全错位掉，虽然你可以说，我就是这么简单粗暴，不过，我还是愿意处理的更精细些。
        :param df: 包含了所有财务数据列（可能是多列）的财务数据，这个是财务数据的df，不是周频数据
        :return: 列被提换成了TTM数据
        """

        def handle_one_period_ttm(row, finance_column_names,finance_date_column_name):

            # 从这只股票的财务数据(df)得到这一行对应的年报的报告期结束日
            end_date = row[finance_date_column_name]
            last_year_same_date = utils.last_year(end_date)  # 去年的同期日
            last_year_end_date = utils.last_year(end_date[:4] + "1231")  # 去年的最后一天

            # 从这只股票的财务数据中，按日期检索出去年年末，和，去年同期的财务指标数据
            df_last_year_same_date = df[df[finance_date_column_name] == last_year_same_date]
            df_last_year_end_date = df[df[finance_date_column_name] == last_year_end_date]

            # 如果去年年末，和，去年同期的财务指标数据都存在，就直接计算
            if len(df_last_year_same_date) == 1 and len(df_last_year_end_date) == 1:
                # 当期TTM指标 = 今年同期 + 年报指标 - 去年同期
                ttm_values = \
                    row[finance_column_names] + \
                    df_last_year_end_date[finance_column_names].iloc[0] - \
                    df_last_year_same_date[finance_column_names].iloc[0]
                # print(row[finance_column_names].tolist())
                # print(df_last_year_end_date[finance_column_names].iloc[0].tolist())
                # print(df_last_year_same_date[finance_column_names].iloc[0].tolist())
                # logger.debug('当期[%s]_TTM:%r',end_date,ttm_values.tolist())
            # 否则，就用当期的近似计算
            else:
                logger.debug("无法获得[%s]去年同期%s[%d条]或者去年年末%s[%d条]信息",
                             row.name,
                             last_year_same_date,
                             len(df_last_year_same_date),
                             last_year_end_date,
                             len(df_last_year_end_date))
                logger.debug("使用[%s]当期[%s]的数据近似模拟",row.name,end_date)
                ttm_values = self.__calculate_ttm_by_same_year_peirod(row[finance_column_names], end_date)
            # 重新替换掉旧的非TTM数据
            row[finance_column_names] = ttm_values
            return row

        # print("=" * 120)
        # print(df)
        return df.apply(handle_one_period_ttm,
                        axis=1,
                        finance_column_names=finance_column_names,
                        finance_date_column_name=finance_date_column_name)
        # print(df2)

    def __calculate_ttm_by_same_year_peirod(self, row, end_date):
        PERIOD_DEF = {
            '0331': 4,
            '0630': 2,
            '0930': 1.33,
            '1231': 1
        }
        periods = PERIOD_DEF.get(end_date[-4:], None)
        if periods is None:
            logger.warning("无法根据[%s]财务日期[%s]得到财务的季度间隔数", row['ts_code'],end_date)
            return np.nan
        return row * periods


# python -m mlstock.factors.mixin.ttm_mixin
if __name__ == '__main__':
    utils.init_logger(file=False)

    start_date = '20150703'
    end_date = '20190826'
    stocks = ['600000.SH', '002357.SZ', '000404.SZ', '600230.SH']
    from mlstock.factors.income import Income
    col_names = Income(None,None).tushare_name

    import tushare as ts
    import pandas as pd

    pro = ts.pro_api()

    df_list = []
    for ts_code in stocks:
        df = pro.income(ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date)
        df_list.append(df)

    df = pd.concat(df_list)
    logger.info("原始数据：\n%r", df)
    df = TTMMixin().ttm(df, col_names)
    logger.info("计算完毕的TTM数据：\n%r", df)
