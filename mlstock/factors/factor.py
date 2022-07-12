import logging
from abc import ABC, abstractmethod

from mfm_learner.datasource import datasource_factory
from mfm_learner.utils import CONF
import pandas as pd
from sqlalchemy import Table, MetaData

from mfm_learner.utils import utils

from mlstock.data.datasource import DataSource

logger = logging.getLogger(__name__)


class Factor(ABC):
    """
    因子，也就是指标
    """

    def __init__(self):
        self.datasource = DataSource()

    # 英文名
    @abstractmethod
    @property
    def name(self):
        return "Unknown"

    # 中文名
    @abstractmethod
    @property
    def cname(self):
        return "未定义"

    @abstractmethod
    def calculate(self, df):
        raise ImportError()
