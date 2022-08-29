function elapse(){
  duration=$SECONDS
  echo "耗时 $(($duration / 60)) 分 $(($duration % 60)) 秒."
}

if [ "$1" != "" ]
then
  MODEL=all
else
  MODEL=$1
fi

echo "准备开始训练..."

DATA_FILE=data/`ls -1rt data/|grep factor|tail -n 1`

echo "  使用最新的数据文件：$DATA_FILE"

SECONDS=0
python -m mlstock.ml.train \
-s 20090101 -e 20190101 \
-t $MODEL \
-d $DATA_FILE >./logs/console.train.log 2>&1
echo "训练结束"
elapse