EALIEST_DATE='20080101'
STOCK_IPO_YEARS = 1 # 至少上市1年的股票
CONF_PATH = "conf/config.yml"
RESERVED_PERIODS = 50 # 预留50周的数据,目前看到的需要最长预留的是MACD:35，但是中间有各种假期、节日啥的，所以，预留40不够，改到50了
CODE_DATE = ['ts_code','trade_date'] # 定义一个最常用的取得数据集的 ts_code和 trade_date 的列名