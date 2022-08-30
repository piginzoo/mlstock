# prepare_factor.sh 100  (调试模式，只加载100只股票)
function elapse(){
  duration=$SECONDS
  echo "耗时 $(($duration / 60)) 分 $(($duration % 60)) 秒."
}

echo "准备开始因子生成..."
SECONDS=0
if [ "$1" != "" ]
then
  echo "[ 调试模式 ]"
  python -m mlstock.ml.prepare_factor -in -s 20080101 -e 20220901 -n $1 >./logs/console.data.log 2>&1
else
  python -m mlstock.ml.prepare_factor -in -s 20080101 -e 20220901 >./logs/console.data.log 2>&1
fi
DATA_FILE=data/`ls -1rt data/|grep .csv|tail -n 1`
echo "因子生成结束，生成的因子文件为：$DATA_FILE"
elapse