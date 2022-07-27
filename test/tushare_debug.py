import pandas as pd
import numpy as np
import tushare as ts

pro = ts.pro_api()
df = pro.query('daily', ts_code='000001.SZ', start_date='20180101', end_date='202001231')

import pdb;pdb.set_trace()