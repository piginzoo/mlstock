# 前言

感谢B站UP主[致敬大神](https://www.bilibili.com/video/BV1564y1b7PR)，这个项目是站在她的华泰金工的研报复现的基础上的一个项目。

她虽然已经给出了完整的代码，但是我更想在她的工作基础上，做成一个可以一键run的机器学习项目，方便更好的高效运行。

也通过对她的代码的重新梳理、理解和调试，更好的理解和掌握机器学习在量化投资中的应用。

这个项目计划会持续2~3个月，可以持续关注。



# 运行这个程序

## 1、安装各种包

```
brew install ta-lib
pip install -r requirement.txt
```
如遇中断，需要手工逐个安装。

## 2.准备数据

这个需要大量的数据，可以参考之前 [tushare下载程序](https://github.com/piginzoo/mfm_learner/tree/main/mfm_learner/utils/tushare_download),
从tushare下载所有的数据到本地数据库，

```
git clone https://github.com/piginzoo/mfm_learner.git
cd mfm_learner
python -m utils.tushare_download.updator
```
整个过程大约会持续3-4个小时，会拉去从2008.1.1-至今的数据，2008年之前的数据由于股改问题，参考价值不大，故未使用。

## 3.一键式完成整个过程

运行pipelne.sh，完成因子清洗、训练、指标、回测：

```shell
bin/pipline.sh
```

整个过程会持续40分钟~1小时。最终会生成 data/plot.jpg，来显示回测结果。

## 4. 单独运行每个环节

### 4.1 因子清洗

参数细节参考：[prepare_factor.py](mlstock/ml/prepare_factor.py)

```python
python -m mlstock.ml.prepare_factor -in -s 20080101 -e 20220901
```
### 4.2 训练

参数细节参考：[train.py](mlstock/ml/train.py)

```python
python -m mlstock.ml.train --train all --data data/processed_industry_neutral_20080101_20220901_20220828152110.csv
```

### 4.3 指标评价

参数细节参考：[evaluate.py](mlstock/ml/evaluate.py)

```python
python -m mlstock.ml.backtest \
-s 20190101 -e 20220901 \
-mp pct_ridge_20220828224004.model \
-mw winloss_xgboost_20220828224019.model \
-d processed_industry_neutral_20080101_20220901_20220828152110.csv
```

### 4.4 回测

参数细节参考：[backtest.py](mlstock/ml/backtest.py)

```python
python -m mlstock.ml.backtest \
-s 20190101 -e 20220901 \
-mp pct_ridge_20220828224004.model \
-mw winloss_xgboost_20220828224019.model \
-d processed_industry_neutral_20080101_20220901_20220828152110.csv
```

# 开发计划

- [X] 实现一个简单的闭环，几个特征，线性回归
- [X] 完善大部分的特征，做特征的数据分布分析，补充异常值，清晰特征
- [X] 尝试多种模型，训练调优，对比效果
- [X] 进一步做分层回测，看模型的实际投资效果
- [ ] 做模型可解释性的实践
- [ ] 尝试深度模型、AlphaNet
- [ ] 确定最优模型，并上实盘

# 开发日志

7.28
- 为了防止macd之类出现nan，预加载了一些日期的数据，目前是观察后，设置为35，加载完再用start_date过滤掉之前的
- 过滤了那些全是nan的财务指标的股票，比如cash_in_subtotal_finance、cash_in_subtotal_invest，4077=>3263，比例20%

8.6
- 实现了ttm、fill处理
- 完善了多种财务指标表的处理：cashflow，balance_sheet，income，fina_indicator，都是对应到tushare的各张财务表
- 实现了特异性波动、beta、alpha等指标

8.13
- 实现了基于fama-french的特异性波动std
- 实现了各种异常值处理，填充

8.28
- 实现了基于"市值+行业"的因子中性化处理
- 实现了基于岭回归的收益率训练代码
- 实现了基于xgboost的涨跌训练代码
- 实现了岭回归回归和xgboost分类的评价指标
- 实现了回测的收益、累计收益和基准的分析

9.2
- 通过逐个因子训练观察，完成了因子的未来函数排查
- 重写了fama-french的特异性波动std，之前的有未来函数问题
- 完善了因子、训练、指标、回测的一键训练，也支持分开运行，更灵活
- 实现了多个回测业务指标
- 

# 数据处理的trick

对于数据，有必要记录一下一些特殊的处理细节
- 先按照daily_basic中的'total_mv','pe_ttm', 'ps_ttm', 'pb'缺失值超过80%，去筛掉了一些股票
- 然后对剩余的股票的daily_basic的上述字段，按照日期向后进行na的填充
- 对财务相关（income,balance_sheet,cashflow,finance_indicator）的数据，都统一除以了实质total_mv，归一化他们
- 和神仔比，我没做行业中性化、PCA，难道是不难，就是不想做了，一个是还要花时间，另外，记得在某集看到，神仔好像后来嫌慢也没做，等有时间我再考虑这事吧
