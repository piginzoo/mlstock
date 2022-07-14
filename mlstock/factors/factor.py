import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Factor(ABC):
    """
    因子，也就是指标
    """

    def __init__(self, datasource):
        self.datasource = datasource

    # 英文名
    @property
    def name(self):
        return "Unknown"

    # 中文名
    @property
    def cname(self):
        return "未定义"

    @abstractmethod
    def calculate(self, df):
        raise ImportError()
