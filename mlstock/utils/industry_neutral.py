import pandas as pd
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin

from mlstock.utils import utils

"""
行业中性处理
    行业市值中性化：
    将填充缺失值后的因子暴露度对行业哑变量和取对数后的市值做线性回归，取残差作为新的因子暴露度。
    数据预处理（下）之中性化:https://www.pianshen.com/article/9590863701/
    其中，Factor_i为股票i的alpha因子
    MktVal_i为股票i的总市值，
    Industry_j,i为行业虚拟变量, 即如果股票i属于行业j则暴露度为1，否则为0，而且每个股票i仅属于一个行业

关于Estimator：
    # https://juejin.cn/post/6844903478788096007
    
    估计器，很多时候可以直接理解成分类器，主要包含两个函数：                
    fit()：训练算法，设置内部参数。接收训练集和类别两个参数。
    predict()：预测测试集类别，参数为测试集。大多数scikit-learn估计器接收和输出的数据格式均为numpy数组或类似格式。
    
    转换器用于数据预处理和数据转换，主要是三个方法：
    fit()：训练算法，设置内部参数。
    transform()：数据转换。
    fit_transform()：合并fit和transform两个方法
    
    使用：
        idst_neutral= IndustryNeutral()
        idst_neutral = idst_neutral.fit(df_train, f_x, f_idst)
        df_train = idst_neutral.transform(df_train, f_x, f_idst)
        df_test = idst_neutral.transform(df_test, f_x, f_idst)
"""


class IndustryMarketNeutral(BaseEstimator, TransformerMixin):
    """
    行业中性化没啥神秘的！他是对一个因子值，比如'资产收益率'，对整个值进行回归，
    它作为Y，X是市值和行业one-hot，回归出来和原值的残差，就是中性化后的因子值。

    用特征数据中的每一列，作为Y，X是行业one-hot和行业市值对数化（我理解是为了小一些），来做一个线性回归模型。
    所以，每一列都会有一个模型，
    然后用每一个模型去预测每个因子的预测值，y_pred，
    然后 y - y_pred 的残差，就是行业+市值中性化后的
    """

    def __init__(self, factor_names, industry_name, market_value_name):
        self.models = {}
        self.factor_names = factor_names
        self.industry_name = industry_name
        self.market_value_name = market_value_name

    def _to_one_hot(self, series):
        pd.get_dummies(series, prefix='industry')

    def _regression_fit(self, X, y):
        # 把异常数据去掉
        X = X[~y.isna()]
        y = y[~y.isna()]
        self.models[y.name] = linear_model.LinearRegression().fit(X, y)
        return

    def _regression_pred(self, X, y):
        pred = self.models[y.name].predict(X)
        return y - pred

    def fit(self, df):
        # 把行业从int变成one-hot，然后再合并上市值信息
        X = pd.concat(
            [df[self.market_value_name],
             self._to_one_hot(df[self.industry_name])],
            axis=1)
        # 每一个特征/因子，即每一列，逐列进行训练，每列训练出一个模型来，保存到字典中
        df[self.factor_names].apply(lambda y: self._regression_fit(X, y))
        return self

    def transform(self, df):
        # 把行业从int变成one-hot，然后再合并上市值信息
        X = pd.concat(
            [df[self.market_value_name],
             self._to_one_hot(df[self.industry_name])],
            axis=1)
        df[self.factor_names] = df[self.factor_names].apply(lambda y: self._regression_pred(X, y))
        return df

# python -m mlstock.utils.industry_neutral
# if __name__ == '__main__':
#     from mlstock.data.datasource import DataSource
#     from mlstock.data.stock_info import StocksInfo
#
#     utils.init_logger(file=False)
#
#     start_date = "20080101"
#     end_date = "20220801"
#     datasource = DataSource()
#     stock_data, ts_codes = load_stock_data(datasource, start_date, end_date, 20)
#
#     from mlstock.factors.macd import MACD
#     factor = MACD(datasource, StocksInfo(ts_codes, start_date, end_date))
#     df_factor = factor.calculate(stock_data)
#     df_weekly = factor.merge(stock_data.df_weekly, df_factor)
#     df_weekly = pd.merge(df_weekly, stock_data.df_daily_basic, how='left', on=['ts_code','trade_date'])
#
#     industry_market_neutral = IndustryMarketNeutral([factor.name],
#                                                     market_value_name='total_mv_log',
#                                                     industry_name='industry')
#     industry_market_neutral.fit(df_weekly)
#     df_weekly = industry_market_neutral.transform(df_weekly)
#     print(df_weekly)