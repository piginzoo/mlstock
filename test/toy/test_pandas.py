import pandas as pd
import numpy as np

def make_dummy_data():
    data1 = [
        ['000001.SZ', '2016-06-21'] + np.random.rand(3).tolist(),
        ['000001.SZ', '2016-06-22'] + np.random.rand(3).tolist(),
        ['000001.SZ', '2016-06-23'] + np.random.rand(3).tolist(),
        ['000001.SZ', '2016-06-24'] + np.random.rand(3).tolist(),
        ['000001.SZ', '2016-06-27'] + np.random.rand(3).tolist(),
        ['000001.SZ', '2016-06-28'] + np.random.rand(3).tolist(),
        ['000002.SH', '2016-06-21'] + np.random.rand(3).tolist(),
        ['000002.SH', '2016-06-22'] + np.random.rand(3).tolist(),
        ['000002.SH', '2016-06-23'] + np.random.rand(3).tolist(),
        ['000002.SH', '2016-06-24'] + np.random.rand(3).tolist(),
        ['000002.SH', '2016-06-27'] + np.random.rand(3).tolist(),
        ['000002.SH', '2016-06-28'] + np.random.rand(3).tolist(),
        ['000002.SH', '2016-06-29'] + np.random.rand(3).tolist(),
        ['000003.SH', '2016-06-18'] + np.random.rand(3).tolist(),
        ['000003.SH', '2016-06-19'] + np.random.rand(3).tolist(),
        ['000003.SH', '2016-06-20'] + np.random.rand(3).tolist(),
        ['000003.SH', '2016-06-21'] + np.random.rand(3).tolist(),
        ['000003.SH', '2016-06-22'] + np.random.rand(3).tolist(),
        ['000003.SH', '2016-06-23'] + np.random.rand(3).tolist(),
        ['000003.SH', '2016-06-24'] + np.random.rand(3).tolist(),
        ['000003.SH', '2016-06-27'] + np.random.rand(3).tolist(),
        ['000003.SH', '2016-06-28'] + np.random.rand(3).tolist()
    ]
    data1 = pd.DataFrame(data1,
                         columns=["ts_code", "datetime", "a1", "a2", "a3"])
    data1['datetime'] = pd.to_datetime(data1['datetime'], format='%Y-%m-%d')  # 时间为日期格式，tushare是str
    return data1


df = make_dummy_data()


def roll_func(s):
    _df = df.loc[s.index][["a1","a2","a3"]]
    # import pdb;pdb.set_trace()
    return _df.sum(axis=0).sum()/(_df.shape[0]*_df.shape[1])


df1 = df.groupby('ts_code').rolling(window=3).mean()
print(df1)
print("-"*80)

df2 = df.groupby('ts_code').rolling(window=3).apply(roll_func, raw=False)
print(df2)

# python -m test.toy.test_pandas
