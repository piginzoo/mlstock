"""
他们说RSI，是个平稳序列，是么？我自己试试。
结论：
    我测了10只左右，大约都在lags=20，也就是延迟在20步的时候，
    开始进入1个方差的范围，也就是趋于平稳了。
    另外，对随机性做了Q检验，发现P值是始终是0的，说明，lag=40以内是自相关性=0的概率是0，
    H0假设：这个是个随机序列，这个事件应该大概率发生，如果小概率发生了，就会被推翻。
    我看到的是，我自己的这个序列机会p-value总是0，说明是小概率，说明原假设被推翻，所以不是一个随机序列。
    引自我的假设检验笔记：https://book.piginzoo.com/quantitative/statistics/test.html
    "你假设了一个参数，然后，你用这个参数去算某一次事件的概率，如果这个概率小于0.05，那说明你的假设不靠谱啊，你的假设下，应该大概率发生才对；现在小概率发生了，说明你的假设不对啊。"
后来又测试了macd的hist，果然不是平稳的
"""
import random
import matplotlib
import matplotlib.pyplot as plt
import talib as ta
from matplotlib.font_manager import FontProperties
from statsmodels.graphics.tsaplots import plot_acf
import sys
import tushare as ts

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号，解决负号'-'显示为方块的问题

token = sys.argv[1]
pro = ts.pro_api(token)


# 接下来，还要看看是不是随机序列：https://mlln.cn/2017/10/26/python%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E5%88%86%E6%9E%90-%E7%BA%AF%E9%9A%8F%E6%9C%BA%E6%80%A7%E6%A3%80%E9%AA%8C/
def boxpierce_test(x, plt):
    from statsmodels.sandbox.stats.diagnostic import acorr_ljungbox
    qljungbox, pval, qboxpierce, pvalbp = acorr_ljungbox(x, boxpierce=True)
    for i in range(len(pval)):
        print('真实数据：qljungbox, pval, qboxpierce, pvalbp:', qljungbox[i], pval[i], qboxpierce[i], pvalbp[i])

    ax = plt.add_subplot(425)
    ax.plot(qljungbox, label='qljungbox');
    ax.set_ylabel('True-Q')
    ax.plot(qboxpierce, label='qboxpierce')
    ax.legend()

    ax = plt.add_subplot(426)
    ax.plot(pval, label='pval');
    ax.set_ylabel('P')
    ax.plot(pvalbp, label='pvalbp')
    ax.legend()

    ax = plt.add_subplot(427)
    x = [random.randint(1, 200) for i in range(len(x))]
    qljungbox, pval, qboxpierce, pvalbp = acorr_ljungbox(x, boxpierce=True)
    ax.plot(qljungbox, label='qljungbox');
    ax.set_ylabel('random-Q')
    ax.plot(qboxpierce, label='qboxpierce')
    ax.legend()

    ax = plt.add_subplot(428)
    ax.plot(pval, label='pval');
    ax.set_ylabel('P')
    ax.plot(pvalbp, label='pvalbp')
    ax.legend()

    print('随机数据：qljungbox, pval, qboxpierce, pvalbp:', qljungbox, pval, qboxpierce, pvalbp)


def test_kpss(df):
    """
    https://juejin.cn/post/7121729587079282696
    :param df:
    :return:
    """
    import statsmodels.api as sm
    test = sm.tsa.stattools.kpss(df, regression='ct')
    print("KPSS平稳性检验结果：",test)
    print("说明：")
    print("\t原假设H0：时间序列是趋势静止的，备择假设H1：时间序列不是趋势静止的")
    print("\t如果p值小于显著性水平α=0.05，就拒绝无效假设，不是趋势静止的。")
    print("-"*40)
    print("T统计量：",test[0])
    print("p-value：", test[1])
    print("lags延迟期数：", test[2])
    print("置信区间下的临界T统计量：")
    print("\t 1% ：", test[3]['1%'])
    print("\t 5% ：", test[3]['5%'])
    print("\t 10% ：", test[3]['10%'])
    print("检验结果（是否平稳）：",
          test[1] > 0.05
    )


def test_acf(df):
    """
    https://cloud.tencent.com/developer/article/1737142
    :param df:
    :return:
    """
    from statsmodels.tsa.stattools import adfuller
    adftest = adfuller(df, autolag='AIC')  # ADF检验
    print("ADF平稳性检验结果：",adftest)
    print("说明：")
    print("\tADF检验的原假设是存在单位根,统计值是小于1%水平下的数字就可以极显著的拒绝原假设，认为数据平稳")
    print("\tADF结果T统计量同时小于1%、5%、10%三个level的统计量,说明平稳")
    print("-"*40)
    print("T统计量：",adftest[0])
    print("p-value：", adftest[1])
    print("lags延迟期数：", adftest[2])
    print("测试的次数：", adftest[3])
    print("置信区间下的临界T统计量：")
    print("\t 1% ：", adftest[4]['1%'])
    print("\t 5% ：", adftest[4]['5%'])
    print("\t 10% ：", adftest[4]['10%'])
    print("检验结果（是否平稳）：",
          adftest[0]<adftest[4]['1%'] and adftest[0]<adftest[4]['5%'] and adftest[0]<adftest[4]['10%']
    )

def test_stock(code):
    df = pro.daily(stock_code=code)
    fig = plt.figure(figsize=(16, 6))

    plt.clf()
    rsi = ta.RSI(df['close'])
    rsi.dropna(inplace=True)

    print("** ADF平稳性检验 **")
    print("=" * 80)

    test_acf(rsi)

    print("** KPSS平稳性检验 **")
    print("="*80)

    test_kpss(rsi)

    print(f"** 画自相关图 debug/平稳性随机性_rsi_{code}.jpg **")
    print("="*80)

    test_stationarity(rsi, fig)
    boxpierce_test(rsi, fig)
    plt.savefig(f"debug/平稳性随机性_rsi_{code}.jpg")

    # plt.clf()
    # macd,macdsignal,macdhist = ta.MACD(df['close'])
    # macdhist.dropna(inplace=True)
    # test_stationarity(macdhist, fig)
    # boxpierce_test(macdhist, fig)
    # plt.savefig(f"debug/平稳性随机性_macd_{code}.jpg")


def test_stationarity(x, fig):
    """换自相关图"""

    font = FontProperties()

    ax = fig.add_subplot(421)
    ax.set_ylabel("return", fontproperties=font, fontsize=16)
    ax.set_yticklabels([str(x * 100) + "0%" for x in ax.get_yticks()], fontproperties=font, fontsize=14)
    ax.set_title("RSI stationarity", fontproperties=font, fontsize=16)
    ax.grid()
    plt.plot(x)

    ax = fig.add_subplot(422)
    plot_acf(x, ax=ax, lags=50)

    from pandas.plotting import autocorrelation_plot
    ax = fig.add_subplot(423)
    autocorrelation_plot(x, ax=ax)


# python -m mlstock.research.test_adf_kpss.py
if __name__ == '__main__':
    for code in ["600495.SH"]: #, "600540.SH", "600819.SH", "600138.SH", "002357.SZ", "002119.SZ"]:
        test_stock(code)