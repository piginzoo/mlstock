import argparse


def handle(df,model,factor_names):
    if 'y_pred' in df.columns:
        logger.info("数据集已经预测过，无需再预测收益率了")
    else:
        df['y_pred'] = df[factor_names].apply(lambda x:model.predict(x), axis=1)
        df['y_pred_rank'] = df.groupby('trade_date').y_pred.rank(ascending=False)

    return df_pct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 数据相关的
    parser.add_argument('-s', '--start_date', type=str, default="20090101", help="开始日期")
    parser.add_argument('-e', '--end_date', type=str, default="20220901", help="结束日期")
    parser.add_argument('-n', '--num', type=int, default=100000, help="股票数量，调试用")
    parser.add_argument('-p', '--preload', type=str, default=None, help="预先加载的因子数据文件的路径，不再从头计算因子")
    parser.add_argument('-in', '--industry_neutral', action='store_true', default=False, help="是否做行业中性处理")

    # 训练相关的
    parser.add_argument('-t', '--train', type=str, default="all", help="all|pct|winloss : 训练所有|仅训练收益|仅训练涨跌")

    # 全局的
    parser.add_argument('-d', '--debug', action='store_true', default=False, help="是否调试")

    args = parser.parse_args()

    if args.debug:
        print("【调试模式】")
        utils.init_logger(file=True, log_level=logging.DEBUG)
    else:
        utils.init_logger(file=True, log_level=logging.INFO)

    main(args)
