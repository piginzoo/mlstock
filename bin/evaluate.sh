function elapse(){
  duration=$SECONDS
  echo "耗时 $(($duration / 60)) 分 $(($duration % 60)) 秒."
}

echo "准备开始指标评测..."

DATA_FILE=data/`ls -1rt data/|grep .csv|tail -n 1`
PCT_MODEL_FILE=model/`ls -1rt model/|grep pct_ridge|tail -n 1`
WINLOSS_MODEL_FILE=model/`ls -1rt model/|grep winloss|tail -n 1`

echo "  使用最新的数据文件：$DATA_FILE"
echo "  使用最新的收益模型：$PCT_MODEL_FILE"
echo "  使用最新的涨跌模型：$WINLOSS_MODEL_FILE"

SECONDS=0
python -m mlstock.ml.evaluate \
-s 20190101 -e 20220901 \
-mp $PCT_MODEL_FILE \
-mw $WINLOSS_MODEL_FILE \
-d $DATA_FILE >./logs/console.evaluate.log 2>&1
echo "指标评测结束"
elapse