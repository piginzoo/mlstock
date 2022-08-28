function elapse(){
  duration=$SECONDS
  echo "耗时 $(($duration / 60)) 分 $(($duration % 60)) 秒."
}

echo "准备开始因子生成..."
SECONDS=0
python -m mlstock.ml.prepare_factor -in -s 20080101 -e 20220901 -n 100 >./logs/console.data.log 2>&1
DATA_FILE=data/`ls -1rt data/|grep .csv|tail -n 1`
echo "因子生成结束，生成的因子文件为：$DATA_FILE"
elapse

echo "准备开始训练(按回车继续...)"
#read -n 1
SECONDS=0
python -m mlstock.ml.train --train all --data $DATA_FILE >./logs/console.train.log 2>&1
PCT_MODEL_FILE=model/`ls -1rt model/|grep pct_ridge|tail -n 1`
WINLOSS_MODEL_FILE=model/`ls -1rt model/|grep winloss|tail -n 1`
echo "训练结束,收益模型为[$PCT_MODEL_FILE],涨跌模型为[$WINLOSS_MODEL_FILE]"
elapse

echo "准备开始指标评测(按回车继续...)"
#read -n 1
SECONDS=0
python -m mlstock.ml.evaluate \
-s 20190101 -e 20220901 \
-mp $PCT_MODEL_FILE \
-mw $WINLOSS_MODEL_FILE \
-d $DATA_FILE >./logs/console.evaluate.log 2>&1
echo "指标评测结束"
elapse

echo "准备开始回测(按回车继续...)"
#read -n 1
SECONDS=0
python -m mlstock.ml.backtest \
-s 20190101 -e 20220901 \
-mp $PCT_MODEL_FILE \
-mw $WINLOSS_MODEL_FILE \
-d $DATA_FILE >./logs/console.backtest.log 2>&1
echo "回测结束"
elapse