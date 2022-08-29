# -----------------------------------------------------------------------------------------------
# 这是一个一站式的训练，我最开始是想做成一体式的，或者叫一键式的，即从因子数据生成、清晰、训练、评测、回测一口气都完成，
# 后来发现，这样会很不灵活，比如我想调试、或者重新生成模型，都变得困难，而且，当跑全数据的时候，整个过程极其漫长，
# 所以，就拆成了目前的几段分别做的：prepare_data.py, train.py, evaluate.py, backtest.py,
# 然后用这个批处理，来串起来，实现我之前想的一键式训练至回测过程。
# -----------------------------------------------------------------------------------------------

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