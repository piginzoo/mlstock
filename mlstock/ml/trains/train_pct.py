import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from mlstock.ml.trains.train_action import TrainAction
from mlstock.utils import utils
from mlstock.utils.utils import time_elapse

logger = logging.getLogger(__name__)


class TrainPct(TrainAction):

    def get_model_name(self):
        return f"pct_ridge_{utils.now()}.model"

    def set_target(self, df_data):
        return df_data  # 啥也不做

    def _train(self, X_train, y_train):
        """用岭回归预测下周收益"""
        best_hyperparam = self.search_best_hyperparams(X_train, y_train)
        ridge = Ridge(alpha=best_hyperparam)
        ridge.fit(X_train, y_train)
        return ridge

    def search_best_hyperparams(self, X_train, y_train):
        """
        超找最好的超参
        :param X_train:
        :param y_train:
        :return:
        """
        # 做这个是为了人肉看一下最好的岭回归的超参alpha的最优值是啥
        # 是没必要的，因为后面还会用 gridsearch自动跑一下，做这个就是想直观的感受一下
        results = {}
        alpha_scope = np.arange(start=1, stop=200, step=10)
        for i in alpha_scope:
            # Ridge和Lasso回归分别代表L1和L2的正则化，L1会把系数压缩到0，而L2则不会，同时L1还有挑选特征的作用
            ridge = Ridge(alpha=i)
            results[i] = cross_val_score(ridge, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean()

        # 按照value排序：{1: 1, 2: 2, 3: 3} =>[(3, 3), (2, 2), (1, 1)]
        sorted_results = sorted(results.items(), key=lambda x: (x[1], x[0]), reverse=True)

        logger.info("超参数/样本和预测的均方误差：%r", results)
        logger.info("最好的超参数为：%.0f, 对应的最好的均方误差的均值：%.2f",
                    sorted_results[0][0],
                    sorted_results[0][1])

        # 保存超参的图像
        fig = plt.figure(figsize=(20, 5))
        plt.title('Best Alpha')
        plt.plot(alpha_scope, [results[i] for i in alpha_scope], c="red", label="alpha")
        plt.legend()
        fig.savefig("data/best_alpha.jpg")

        # 自动搜索，暂时保留，上面的代码中，我手工搜索了，还画了图
        # # 用grid search找最好的alpha：[200,205,...,500]
        # # grid search的参数是alpha，岭回归就这样一个参数，用于约束参数的平方和
        # # grid search的入参包括alpha的范围，K-Fold的折数(cv)，还有岭回归评价的函数(负均方误差)
        # grid_search = GridSearchCV(Ridge(),
        #                            {'alpha': alpha_scope},
        #                            cv=5,  # 5折(KFold值)
        #                            scoring='neg_mean_squared_error')
        # grid_search.fit(X_train, y_train)
        # # model = grid_search.estimator.fit(X_train, y_train)
        # logger.info("GridSarch最好的成绩:%.5f", grid_search.best_score_)
        # # 得到的结果是495，确实和上面人肉跑是一样的结果
        # logger.info("GridSarch最好的参数:%.5f", grid_search.best_estimator_.alpha)

        best_hyperparam = sorted_results[0][0]
        return best_hyperparam
