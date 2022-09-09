import numpy as np
from pandas import DataFrame
import talib
df = df_weekly_index = DataFrame()


def macd():
    macd = talib.MACD(df_weekly_index.close)

def ma(df):
    df['SMA5'] = talib.SMA(np.array(df.close), timeperiod=5)
    df['SMA10'] = talib.SMA(np.array(df.close), timeperiod=10)
    df['transaction'] = df.SMA5>=df.SMA10
    return df[['trade_date','transaction']]

def kdj(df):
    df['K'], df['D'] = talib.STOCH(df['high'].values,
                                   df['low'].values,
                                   df['close'].values,
                                   fastk_period=9,
                                   slowk_period=3,
                                   slowk_matype=0,
                                   slowd_period=3,
                                  slowd_matype=0)
    ####处理数据，计算J值
    df['K'].fillna(0,inplace=True)
    df['D'].fillna(0,inplace=True)
    df['J']=3*df['K']-2*df['D']

    ####计算金叉和死叉
    df['KDJ_金叉死叉'] = ''
    ####令K>D 为真值
    kdj_position=df['K']>df['D']

    ####当Ki>Di 为真，Ki-1 <Di-1 为假 为金叉
    df.loc[kdj_position[(kdj_position == True) & (kdj_position.shift() == False)].index, 'KDJ_金叉死叉'] = '金叉'

    ####当Ki<Di 为真，Ki-1 >Di-1 为假 为死叉
    df.loc[kdj_position[(kdj_position == False) & (kdj_position.shift() == True)].index, 'KDJ_金叉死叉'] = '死叉'