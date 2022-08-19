echo "训练..."
nohup python -m mlstock.ml.train -d >./logs/console.log 2>&1 &