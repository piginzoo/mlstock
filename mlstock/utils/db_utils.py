import logging
import time

import sqlalchemy
from sqlalchemy import create_engine

from mlstock.utils.utils import CONF

logger = logging.getLogger(__name__)

EALIEST_DATE = '20080101'  # 最早的数据日期

def connect_db():
    """
    # https://stackoverflow.com/questions/8645250/how-to-close-sqlalchemy-connection-in-mysql:
        Engine is a factory for connections as well as a ** pool ** of connections, not the connection itself.
        When you say conn.close(), the connection is returned to the connection pool within the Engine,
        not actually closed.
    """

    uid = CONF['database']['uid']
    pwd = CONF['database']['pwd']
    db = CONF['database']['db']
    host = CONF['database']['host']
    port = CONF['database']['port']
    engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}?charset={}".format(uid, pwd, host, port, db, 'utf8'))
    # engine = create_engine('sqlite:///' + DB_FILE + '?check_same_thread=False', echo=echo)  # 是否显示SQL：, echo=True)
    return engine


def is_table_exist(engine, name):
    return sqlalchemy.inspect(engine).has_table(name)


def is_table_index_exist(engine, name):
    if not is_table_exist(engine, name):
        return False

    indices = sqlalchemy.inspect(engine).get_indexes(name)
    return indices and len(indices) > 0


def run_sql(engine, sql):
    c = engine.connect()
    sql = (sql)
    result = c.execute(sql)
    return result


def list_to_sql_format(_list):
    """
    把list转成sql中in要求的格式
    ['a','b','c'] => " 'a','b','c' "
    """
    if type(_list) != list: _list = [_list]
    data = ["\'" + one + "\'" for one in _list]
    return ','.join(data)


def create_db_index(engine, table_name, df):
    if is_table_index_exist(engine, table_name): return

    # 创建索引，需要单的sql处理
    index_sql = None
    if "ts_code" in df.columns and "trade_date" in df.columns:
        index_sql = "create index {}_code_date on {} (ts_code,trade_date);".format(table_name, table_name)
    if "ts_code" in df.columns and "ann_date" in df.columns:
        index_sql = "create index {}_code_date on {} (ts_code,ann_date);".format(table_name, table_name)

    if not index_sql: return

    start_time = time.time()
    engine.execute(index_sql)
    logger.debug("在表[%s]上创建索引，耗时: %.2f %s", table_name, time.time() - start_time, index_sql)
