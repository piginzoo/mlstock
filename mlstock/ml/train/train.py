import logging
import os.path
import time

import joblib
from sklearn.model_selection import train_test_split

from mlstock.const import TRAIN_TEST_SPLIT_DATE
from mlstock.utils.utils import time_elapse

logger = logging.getLogger(__name__)


class Train:

    def __init__(self, factor_names):
        self.factor_names = factor_names

    def set_target(self):
        raise NotImplemented()

    def evaluate(self):
        raise NotImplemented()

    def _train(self, X_train, y_train):
        raise NotImplemented()

    def train(self, df_weekly):
        # 根据子类，来调整target（分类要变成0:1)
        df_weekly = self.set_target(df_weekly)

        # 划分训练集和测试集，测试集占总数据的15%，随机种子为10(如果不定义，会每次都不一样）
        # 2009.1~2022.8,165个月，Test比例0.3，大约是2019.1~2022.8，正好合适
        df_train = df_weekly[df_weekly.trade_date < TRAIN_TEST_SPLIT_DATE]
        df_test = df_weekly[df_weekly.trade_date >= TRAIN_TEST_SPLIT_DATE]
        X_train = df_train[self.factor_names].values
        y_train = df_train.target

        # 训练
        start_time = time.time()
        model = self._train(X_train, y_train)
        self.save_model(model)
        time_elapse(start_time, "⭐️ 岭回归训练完成")

        return model, df_train, df_test

    def get_model_name(self):
        raise NotImplemented()

    def save_model(self, model):
        if not os.path.exists("./model"): os.mkdir("./model")
        model_file_path = f"./model/{self.get_model_name()}"
        joblib.dump(model, model_file_path)
        logger.info("训练结果保存到：%s", model_file_path)
