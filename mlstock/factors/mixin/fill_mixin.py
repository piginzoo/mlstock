class FillMixin:
    def fill(self,df,df_finance):
        """
        将离散的单个的财务TTM信息，反向填充到周频的交易数据里去，
        比如周频，日期为7.22（周五），那对应的财务数据是6.30日的，
        所以要用6.30的TTM数据去填充到这个7.22日上去。
        :param df: df是原始的周频数据，以周五的日期为准
        :param df_finance: 财务数据，只包含财务公告日期，要求之前已经做了ttm数据
        """
        pass