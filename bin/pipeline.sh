echo "准备开始生成因子数据..."
python -m mlstock.ml.prepare_factor -in -s 20080101 -e 20220901 -n 100 >./logs/console.data.log 2>&1

DATA_FILE=data/`ls -1rt data/|grep .csv|tail -n 1`
echo "生成因子数据结束：$DATA_FILE"
echo "准备开始训练..."
read -n 1

python -m mlstock.ml.train --train all --data $DATA_FILE >./logs/console.train.log 2>&1
PCT_MODEL_FILE=model/`ls -1rt model/|grep pct_ridge|tail -n 1`
WINLOSS_MODEL_FILE=model/`ls -1rt model/|grep winloss|tail -n 1`
echo "训练结束,收益模型[$PCT_MODEL_FILE],涨跌模型[$WINLOSS_MODEL_FILE]"
read -n 1

echo "准备开始评测指标..."
python -m mlstock.ml.evaluate \
-s 20190101 -e 20220901 \
-mp $PCT_MODEL_FILE \
-mw $WINLOSS_MODEL_FILE \
-d $DATA_FILE >./logs/console.evaluate.log 2>&1
echo "准备开始评测指标..."

echo "准备开始回测..."
python -m mlstock.ml.backtest \
-s 20190101 -e 20220901 \
-mp $PCT_MODEL_FILE \
-mw $WINLOSS_MODEL_FILE \
-d $DATA_FILE >./logs/console.backtest.log 2>&1
echo "回测结束"