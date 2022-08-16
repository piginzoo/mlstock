
# 前言

感谢B站UP主[致敬大神](https://www.bilibili.com/video/BV1564y1b7PR)，这个项目是站在她的华泰金工的研报复现的基础上的一个项目。

她虽然已经给出了完整的代码，但是我更想在她的工作基础上，做成一个可以一键run的机器学习项目，方便更好的高效运行。

也通过对她的代码的重新梳理、理解和调试，更好的理解和掌握机器学习在量化投资中的应用。

这个项目计划会持续2~3个月，可以持续关注。

# 开发计划

- [X] 实现一个简单的闭环，几个特征，线性回归
- [X] 完善大部分的特征，做特征的数据分布分析，补充异常值，清晰特征
- [ ] 尝试多种模型，训练调优，对比效果
- [ ] 进一步做分层回测，看模型的实际投资效果
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

# 数据处理的trick

对于数据，有必要记录一下一些特殊的处理细节
- 先按照daily_basic中的'total_mv','pe_ttm', 'ps_ttm', 'pb'缺失值超过80%，去筛掉了一些股票
- 然后对剩余的股票的daily_basic的上述字段，按照日期向后进行na的填充
- 对财务相关（income,balance_sheet,cashflow,finance_indicator）的数据，都统一除以了实质total_mv，归一化他们
- 