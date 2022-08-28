import pandas as pd
import numpy as np
import tushare as ts

pro = ts.pro_api()
df = pro.query('daily', ts_code='000001.SZ,000002.SZ,000003.SZ', start_date='20180101', end_date='20201231')

import pdb;pdb.set_trace()