"""
分层回测

在每个一级行业内部对所有个股按因子大小进行排序，每个行业内均分成M个分层组合。

2011-01-31 至 2020-12-31 12*10=120期

设想：
1、跑2009.1~2022.8的数据，训练，先试试cv，然后再试试tscv
2、用模型，整个做2017.1~2022.8做回测


"""


df_result['industry'] = df_test.industry # 并列的默认使用排名均值
df_result['y_rank_idst'] = df_result.groupby('industry').y_pred.rank(ascending=False)

df_result1 = pd.read_csv(raw_data_path4+'/'+result_name+'.csv')
df_result1 = df_result1.sort_values(by=['year_month', 'class_label', 'y_rank_idst', 'industry', 'ts_code']).reset_index(drop=True)
df_result['t_hs300'] = df_result['t_hs300'] * 100
df_result1.groupby(['year_month','class_label'])[['t_pct_chg', 't_hs300']].mean()
df_result1['year'] = df_result1.year_month.apply(lambda x: str(x)[:4])

# 每个月 每个class的收益 = 当月 该class的所有股票 取均值
dfa = df_result1.groupby(['year', 'year_month','class_label'])[['t_pct_chg', 't_hs300', 'y']].mean()
dfa.reset_index(inplace=True)

# 累计收益率
dfb = dfa.groupby('class_label')['t_pct_chg', 't_hs300'].apply(lambda x: (x+1).cumprod()-1)
dfb.columns = ['c_pct_chg', 'c_hs_300']
dfa = dfa.join(dfb)

# 这里重新排序后 需要重置一下索引
dfa = dfa.sort_values(by=['class_label', 'year', 'year_month']).reset_index(drop=True)

i = 1
x = dfa[dfa['class_label']==i].year_month.values
x = [str(x) for x in x]
y1 = dfa[dfa['class_label']==i].t_pct_chg.values
y2 = dfa[dfa['class_label']==i].c_pct_chg.values
label_x = '年月'
label_y1 = '月收益率'
label_y2 = '累计收益率'
color_y1 = '#2A9CAD'
color_y2 = "#FAB03D"
title = '组合{}月收益率及累积收益率'.format(i)
label_y1 = '组合{}月收益率'.format(i)
label_y2 = '组合{}累积收益率'.format(i)
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
plt.xticks(rotation=60)
ax2 = ax1.twinx()  # 做镜像处理
lns1 = ax1.bar(x=x, height=y1, label=label_y1, color=color_y1, alpha=0.7)
lns2 = ax2.plot(x, y2, color=color_y2, ms=10, label=label_y2)
ax1.set_xlabel(label_x)  # 设置x轴标题
ax1.set_ylabel(label_y1)  # 设置Y1轴标题
ax2.set_ylabel(label_y2)  # 设置Y2轴标题
ax1.grid(False)
ax2.grid(False)
# 设置横轴显示
tick_spacing = 6   # 设置密度，比如横坐标9个，设置这个为3,到时候横坐标上就显示 9/3=3个横坐标，
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# 添加标签
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# 添加标题
plt.title(title)
# 背景网格
plt.grid(axis="y")
# 保存图片
save_path = raw_data_path4+'/'+result_name+'_' + title +'.png'
plt.savefig(save_path)
plt.show()


performence = {'投资组合':[],
               '年化收益率':[],
               '年化波动率':[],
               '夏普比率':[],
               '最大回撤':[],
               '年化超额收益率':[],
               '年化跟踪误差':[],
               '信息比率':[],
               '相对基准月胜率':[],
               '超额收益最大回撤':[]
              }
perform['投资组合'] = []
for i in range(M):
    perform['投资组合'].append('组合{}'.format(i+1))
perform['投资组合'].append('沪深300')
perform['投资组合'].append('多空组合')

# 最终的的收益率 + 1 开根号 月份数/12 为年的数量 -1 得到年化收益率
perform['年化收益率'] = []
for i in range(M):
    dfd = dfa[dfa['class_label'] == (i + 1)]
    total_r = dfd.c_pct_chg.values[-1] + 1
    total_m = dfd.shape[0]
    # 年化收益率
    perform['年化收益率'].append(np.power(total_r, 12 / total_m) - 1)

# 沪深300
dfd = dfa[dfa['class_label'] == 1]
total_r = dfd.c_hs_300.values[-1] + 1
total_m = dfd.shape[0]
perform['年化收益率'].append(np.power(total_r, 12 / total_m) - 1)

# 多空组合
dfd = dfa[dfa['class_label'] == 1]
dfe = dfa[dfa['class_label'] == 5]
total_r = dfd.c_pct_chg.values[-1] - dfe.c_pct_chg.values[-1] + 1
total_m = dfd.shape[0]
perform['年化收益率'].append(np.power(total_r, 12 / total_m) - 1)

# 方差/年数 年数=月份数/12
perform_idx = '年化波动率'
perform[perform_idx] = []
for i in range(M):
    dfd = dfa[dfa['class_label'] == (i + 1)]
    total_r = dfd.t_pct_chg.std()  # 使用当月的收益率
    total_m = dfd.shape[0]
    perform[perform_idx].append(total_r * 12 / total_m)


### 年化波动率

# 沪深300
dfd = dfa[dfa['class_label'] == 1]
total_r = dfd.c_hs_300.std()
total_m = dfd.shape[0]
perform[perform_idx].append(total_r * 12 / total_m)
# 多空组合
dfd = dfa[dfa['class_label'] == 1].reset_index(drop=True)
dfe = dfa[dfa['class_label'] == 5].reset_index(drop=True)
total_r = (dfd.t_pct_chg - dfe.t_pct_chg).std()  # 注意这里使用的是当月的收益率
total_m = dfd.shape[0]
perform[perform_idx].append(total_r * 12 / total_m)


### 夏普比率

# 收益均值-无风险收益率 / 收益方差
perform_idx = '夏普比率'
perform[perform_idx]=[]
for i in range(M):
    dfd = dfa[dfa['class_label']==(i+1)]
    total_r = dfd.t_pct_chg.mean() - dfd.t_hs300.mean()
    total_std = dfd.t_pct_chg.std()
    perform[perform_idx].append(total_r/total_std)
# 沪深300
dfd = dfa[dfa['class_label']==1]
total_r = dfd.t_hs300.mean() - dfd.t_hs300.mean()
total_std = dfd.t_hs300.std()
perform[perform_idx].append(total_r/total_std)

# 多空组合
dfd = dfa[dfa['class_label']==1].reset_index(drop=True)
dfe = dfa[dfa['class_label']==5].reset_index(drop=True)
total_r = (dfd.t_pct_chg - dfe.t_pct_chg).mean() - dfd.t_hs300.mean()# 注意这里使用的是当月的收益率
total_std = (dfd.t_pct_chg - dfe.t_pct_chg).std()
perform[perform_idx].append(total_r/total_std)

### 最大回撤


import numpy as np
import matplotlib.pyplot as plt
def MaxDrawdown(return_list):
    '''最大回撤率'''
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    return (return_list[j] - return_list[i]) / (return_list[j])

# 那就是每个   max(历史上max（1+累积收益率）- (1+累积收益率) )
perform_idx = '最大回撤'
perform[perform_idx]=[]
for i in range(M):
    dfd = dfa[dfa['class_label']==(i+1)]
    r_list = dfd.c_pct_chg.values + 1# 需要传入数组 不然series自带的索引 算出的i不对
    perform[perform_idx].append(MaxDrawdown(r_list))

# 沪深300
dfd = dfa[dfa['class_label']==1]
r_list = dfd.c_hs_300.values + 1
perform[perform_idx].append(MaxDrawdown(r_list))

# 多空组合
dfd = dfa[dfa['class_label']==1].reset_index(drop=True)
dfe = dfa[dfa['class_label']==5].reset_index(drop=True)
r_list = (dfd.c_pct_chg - dfe.c_pct_chg).values + 1
perform[perform_idx].append(MaxDrawdown(r_list))