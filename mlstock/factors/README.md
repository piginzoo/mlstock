# 因子说明

这里列出所有， 《华泰人工智能系列》使用的因子，神仔也是参考了这个list，不过她的有所增加，参考《Jupiter 0102_数据整理》。

# 华泰人工智能系列使用的因子（48个特征）

| 实现               |   大类因子  |  因子描述                  | 因子方向 | 
|-------------------|------------|----------------------------|-------| 
| 估值               | EP         | 净利润（TTM）/总市值          | 1    |                    
| 估值               | EPcut      | 扣除非经常性损益后净利润（TTM）/总市值 | 1    |                    
| 估值               | BP         | 净资产/总市值               | 1    |                    
| 估值               | SP         | 营业收入（TTM）/总市值         | 1    |                    
| 估值               | NCFP       | 净现金流（TTM）/总市值         | 1    |                    
| 估值               | OCFP       | 经营性现金流（TTM）/总市值       | 1    |                    
| 估值               | DP         | 近12 个月现金红利（按除息日计）/总市值 | 1    |
| 估值               | G/PE       | 净利润（TTM）同比增长率/PE_TTM  | 1    |                    
| 成长               | Sales_G_q  | 营业收入（最新财报，YTD）同比增长率   | 1    |
| 成长               | Prof it_G_q | 净利润（最新财报，YTD）同比增长率    | 1    |
| 成长               | OCF_G_q    | 经营性现金流（最新财报，YTD）同比增长率 | 1    |
| 成长               | ROE_G_q ROE | （最新财报，YTD）同比增长率       | 1    |
| 财务质量            | ROE_q ROE  | （最新财报，YTD）            | 1    |
| 财务质量            | ROE_ttm ROE | （最新财报，TTM）            | 1    |
| 财务质量            | ROA_q ROA  | （最新财报，YTD）            | 1    |
| 财务质量            | ROA_ttm ROA | （最新财报，TTM）            | 1    |
| 财务质量            | grossprofitmargin_q | 毛利率（最新财报，YTD）         | 1    |
| 财务质量            | grossprofitmargin_ttm | 毛利率（最新财报，TTM）         | 1    |
| 财务质量            | prof itmargin_q | 扣除非经常性损益后净利润率（最新财报，YTD） | 1    |
| 财务质量            | prof itmargin_ttm | 扣除非经常性损益后净利润率（最新财报，TTM） | 1    |
| 财务质量            | assetturnover_q | 资产周转率（最新财报，YTD）       | 1    |
| 财务质量            | assetturnover_ttm | 资产周转率（最新财报，TTM）       | 1    |
| 财务质量            | operationcashf lowratio_q | 经营性现金流/净利润（最新财报，YTD）  | 1    |
| 财务质量            | operationcashf lowratio_ttm | 经营性现金流/净利润（最新财报，TTM）  | 1    |
| 杠杆               | financial_leverage | 总资产/净资产               | -1   |
| 杠杆               | debtequityratio | 非流动负债/净资产             | -1   |
| 杠杆               | cashratio  | 现金比率                  | 1    |
| 杠杆               | currentratio | 流动比率                  | 1    |
| 市值               | ln_capital | 总市值取对数                | -1   |
| 动量反转            | HAlpha     | 个股 60 个月收益与上证综指回归的截距项 | -1   |
| 动量反转            | return_Nm  | 个股最近N 个月收益率，N=1，3，6，12 | -1   |
| 动量反转            | wgt_return_Nm | 个股最近N 个月内用每日换手率乘以每日收益率求算术平均值，N=1，3，6，12 | -1   |
| 动量反转            | exp_w gt_return_Nm | 个股最近N 个月内用每日换手率乘以函数exp(-x_i/N/4)再乘以每日收益率求算术平均值，x_i 为该日距离截面日的交易日的个数，N=1，3，6，12 | -1   |
| 波动率              | std_FF3factor_Nm | 特质波动率——个股最近N个月内用日频收益率对Fama French 三因子回归的残差的标准差，N=1，3，6，12 | -1   |
| 波动率              | std_Nm     | 个股最近N个月的日收益率序列标准差，N=1，3，6， 12 | -1   |
| 股价                | ln_price   | 股价取对数                 | -1   |
| beta               | beta       | 个股 60 个月收益与上证综指回归的beta | -1   |
| 换手率              | turn_Nm    | 个股最近N个月内日均换手率（剔除停牌、涨跌停的交易日），N=1，3，6，12 | -1   |
| 换手率              | bias_turn_Nm | 个股最近N 个月内日均换手率除以最近2 年内日均换手率（剔除停牌、涨跌停的交易日）再减去1，N=1，3，6，12 | -1   |
| 情绪               | rating_average | wind 评级的平均值           | 1    |
| 情绪               | rating_change | wind 评级（上调家数-下调家数）/总数 | 1    |
| 情绪               | rating_targetprice | wind 一致目标价/现价-1       | 1    |
| 股东               | holder_avgpctchange | 户均持股比例的同比增长率          | 1    |
| 技术               | MACD       | 经典技术指标（释义可参考百度百科），长周期取30日，短周期取10 日，计算DEA 均线的周期（中周期）取15 日 | -1   |
| 技术               | DEA        | | -1   |
| 技术               | DIF        | | -1   |
| 技术               | RSI        | 经典技术指标，周期取20 日        | -1   |
| 技术               | PSY        | 经典技术指标，周期取20 日        | -1   |
| 技术               | BIAS       | 经典技术指标，周期取20 日        | -1   |

# 神仔整理的因子（96个特征）
```
['ts_code', 'year_month', 'macd', 'dea', 'dif', 'rsi', 'psy', 'bias',
       'close_hfq', 'close_hfq_log', 'pct_chg', 'industry', 'list_date',
       'pct_chg_hs300', 'bias_turn_1m', 'bias_turn_3m', 'bias_turn_6m',
       'bias_turn_12m', 'total_cur_assets', 'total_nca', 'total_assets',
       'total_cur_liab', 'total_ncl', 'total_liab', 'c_fr_sale_sg',
       'c_inf_fr_operate_a', 'c_paid_goods_s', 'c_paid_to_for_empl',
       'st_cash_out_act', 'stot_inflows_inv_act', 'stot_out_inv_act',
       'n_cashflow_inv_act', 'stot_cash_in_fnc_act', 'stot_cashout_fnc_act',
       'n_cash_flows_fnc_act', 'trade_date', 'close_wfq', 'turnover_rate',
       'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
       'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv',
       'total_mv_log', 'esg', 'exp_wgt_return_20d', 'exp_wgt_return_60d',
       'exp_wgt_return_120d', 'exp_wgt_return_240d', 'ar_turn', 'ca_turn',
       'fa_turn', 'assets_turn', 'current_ratio', 'quick_ratio',
       'ocf_to_shortdebt', 'debt_to_eqt', 'tangibleasset_to_debt',
       'profit_to_op', 'roa_yearly', 'tr_yoy', 'or_yoy', 'ebt_yoy', 'op_yoy',
       'HAlpha', 'Beta', 'holder_chg', 'basic_eps', 'diluted_eps',
       'total_revenue', 'total_cogs', 'operate_profit', 'non_oper_income',
       'non_oper_exp', 'total_profit', 'n_income', 'return_3m', 'return_6m',
       'return_12m', 'std_20d', 'std_60d', 'std_120d', 'std_240d',
       'turnover_1m', 'turnover_3m', 'turnover_6m', 'turnover_12m',
       'wgt_return_1m', 'wgt_return_3m', 'wgt_return_6m', 'wgt_return_12m']
```