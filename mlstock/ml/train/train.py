import logging
import os.path
import time

import joblib
from sklearn.model_selection import train_test_split

from mlstock.utils.utils import time_elapse

logger = logging.getLogger(__name__)


class Train:

    def set_target(self):
        raise NotImplemented()

    def evaluate(self):
        raise NotImplemented()

    def _train(self, X_train, y_train):
        raise NotImplemented()

    def train(self, df_features, factor_names):
        # 根据子类，来调整target（分类要变成0:1)
        df_features = self.set_target(df_features)

        # 准备训练用数据，需要numpy类型
        X_train = df_features[factor_names].values
        y_train = df_features.target

        # 划分训练集和测试集，测试集占总数据的15%，随机种子为10(如果不定义，会每次都不一样）
        # 2009.1~2022.8,165个月，Test比例0.3，大约是2019.1~2022.8，正好合适
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=10)

        # 训练
        start_time = time.time()
        model = self._train(X_train, y_train)
        self.save_model(model)
        time_elapse(start_time, "⭐️ 岭回归训练完成")

    def get_model_name(self):
        raise NotImplemented()

    def save_model(self, model):
        if not os.path.exists("./model"): os.mkdir("./model")
        model_file_path = f"./model/{self.get_model_name()}"
        joblib.dump(model, model_file_path)
        logger.info("训练结果保存到：%s", model_file_path)