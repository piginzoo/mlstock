import logging
import numpy as np

from mlstock.utils import utils

logger = logging.getLogger(__name__)


class TTMProcessorMixin:
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
    这个是2015.8.15号发布的，所以，我们认为它是2015年的半年报，
    现在我们要计算这条的TTM，也就是20150815发布的信息的TTM。
    公式是： 2015半年报 + 2014年报 - 2014半年报
    所以，我们要找到2014的年报，和，2014的半年报，这个需要去数据里去检索，
    如果找不到这2个数据中的任何一个，就只能用当前的半年报来近似了。
    ========
    注： TTM后，还需要填充处理，参见<FillMixin>
    """

    def ttm(self, df):
        """
        :param df: 包含了 ts_code, ann_date，<各种需要TTM处理的财务指标列> 的 dataframe
        :return: 财务指标列被替换成了TTM值
        """
        # 删除重复值
        df = df[~df[['ts_code', 'ann_date']].duplicated(keep='last')]

        # 剔除Nan
        df.dropna(inplace=True)

        # 对时间，升序排列
        df.sort_values('ann_date', inplace=True)

        return df.groupby('ts_code').apply(self.handle_one_stock_ttm)

    def handle_one_stock_ttm(self, df):
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

        def handle_one_period_ttm(row):
            # print(df)
            print("-" * 80)
            # 如果这条财务数据是年报数据
            if row['ann_date'].endswith("1231"):
                # 直接用这条数据了
                # print(row)
                value = row
                # logger.debug("财务日[%s]是年报数据，使用年报指标[%.2f]作为当日指标", finance_date, value)
            else:
                # 如果回溯到1季报、半年报、3季报，就用其 + 去年的年报 - 去年起对应的xxx报的数据，这样粗暴的公式，是为了简单
                last_year_value = self.__last_year_value(df,
                                                         col_name_finance_date,
                                                         col_name_value,
                                                         finance_date)
                last_year_same_period_value = __last_year_period_value(df_stock_finance, col_name_finance_date,
                                                                       col_name_value, finance_date)
                # 如果去年年报数据为空，或者，也找不到去年的同期的数据，
                if last_year_value is None or last_year_same_period_value is None:
                    value = __calculate_ttm_by_peirod(current_period_value, finance_date)
                    # logger.debug("财务日[%s]是非年报数据，无去年报指标，使用N倍当前指标[%.2f]作为当日指标", finance_date, value)
                else:
                    # 当日指标 = 今年同期 + 年报指标 - 去年同期
                    value = current_period_value + last_year_value - last_year_same_period_value
                    # logger.debug("财务日[%s]是非年报数据，今年同期[%.2f]+年报指标[%.2f]-去年同期[%.2f]=[%.2f]作为当日指标",
                    #              finance_date,
                    #              current_period_value,
                    #              last_year_value,
                    #              last_year_same_period_value,
                    #              value)

        print("=" * 120)
        # print(df)
        df.apply(handle_one_period_ttm, axis=1)

    def __last_year_value(self, df_stock_finance, finance_date_col_name, value_col_name, current_finance_date):
        last_year_finance_date = current_finance_date[:4] + "1231"
        return self.__last_year_period_value(df_stock_finance,
                                             finance_date_col_name,
                                             value_col_name,
                                             current_finance_date=last_year_finance_date)

    def __last_year_period_value(self, df_stock_finance, finance_date_col_name, value_col_name, current_finance_date):
        """获得去年同时期的财务指标"""

        # 获得去年财务年报的时间，20211030=>20201030
        last_year_finance_date = utils.last_year(current_finance_date)
        df = df_stock_finance[df_stock_finance[finance_date_col_name] == last_year_finance_date]
        # assert len(df) == 0 or len(df) == 1, str(df)
        if len(df) == 1: return df[value_col_name].item()
        if len(df) == 0: return None
        logger.warning("记录数超过2条，取第一条：%r", df)
        return df[value_col_name].iloc[0]

    def __calculate_ttm_by_peirod(self, current_period_value, finance_date):
        PERIOD_DEF = {
            '0331': 4,
            '0630': 2,
            '0930': 1.33,
        }

        periods = PERIOD_DEF.get(finance_date[-4:], None)
        if periods is None:
            logger.warning("无法根据财务日期[%s]得到财务的季度间隔数", finance_date)
            return np.nan
        return current_period_value * periods


# python -m mlstock.factors.mixin.ttm_mixin
if __name__ == '__main__':
    utils.init_logger(file=False)

    start_date = '20150703'
    end_date = '20190826'
    stocks = ['600000.SH', '002357.SZ', '000404.SZ', '600230.SH']

    import tushare as ts
    import pandas as pd

    pro = ts.pro_api()

    df_list = []
    for ts_code in stocks:
        df = pro.income(ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,basic_eps,diluted_eps')
        df_list.append(df)

    df = pd.concat(df_list)

    TTMProcessorMixin().ttm(df)
