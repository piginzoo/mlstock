import logging

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from mlstock.ml.trains.train_action import TrainAction
from mlstock.utils import utils

logger = logging.getLogger(__name__)

PARAMS_MODE = 'fix'  # fix | optimize


class TrainWinLoss(TrainAction):

    def get_model_name(self):
        return f"winloss_xgboost_{utils.now()}.model"

    def set_target(self, df_data):
        df_data['target'] = df_data.target.apply(lambda x: 1 if x > 0 else 0)
        logger.info("设置target为分类：0跌，1涨")
        return df_data

    def _train(self, X_train, y_train):
        """
        Xgboost来做输赢判断，参考：https://cloud.tencent.com/developer/article/1656126
        :return:
        """
        # https://so.muouseo.com/qa/em6w1x8w20k8.html
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

        if PARAMS_MODE == 'optimize':
            # 创建xgb分类模型实例
            model = XGBClassifier(nthread=1)

            # 待搜索的参数列表空间
            param_list = {"max_depth": [3, 5, 7, 9],
                          "n_estimators": [10, 30, 50, 70, 90]}

            # 创建网格搜索
            grid_search = GridSearchCV(model,
                                       param_grid=param_list,
                                       cv=5,
                                       verbose=10,
                                       scoring='f1_weighted',  # TODO:f1???
                                       n_jobs=15)  # 最多15个进程同时跑: 1个进程2G内存，15x2=30G内存使用，不能再多了
            # 基于flights数据集执行搜索
            grid_search.fit(X_train, y_train)

            # 输出搜索结果
            logger.debug("GridSearch出最优参数：%r", grid_search.best_estimator_)

            return grid_search.best_estimator_
        else:
            # 创建xgb分类模型实例
            # 这个参数是由上面的优化结果得出的，上面的时不时跑一次，然后把最优结果抄到这里
            model = XGBClassifier(nthread=1, max_depth=7, n_estimators=50)
            model.fit(X_train, y_train)
            return model
