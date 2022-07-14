from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


def standarlize(df):
    scaler = StandardScaler() # z = (x - u) / sigma
    scaler.fit(df_train[f_x])
    df_train[f_x] = scaler.transform(df_train[f_x])
    df_test[f_x] = scaler.transform(df_test[f_x])


def train(X, y):
    reg = linear_model.LinearRegression()
    model = reg.fit(X, y)


def pred(X):
    y = model.predict(X)
    return y


def evaluate(df):
    # 测试标签y和预测y_pred相关性，到底准不准啊
    df[['y', 'y_pred']].corr().iloc[0, 1]

    # 看一下rank的IC，不看值相关性，而是看排名的相关性
    df['y_rank'] = df.y.rank(ascending=False)  # 并列的默认使用排名均值
    df['y_pred_rank'] = df.y_pred.rank(ascending=False)
    df[['y_rank', 'y_pred_rank']].corr().iloc[0,1]

def load_data():
    return X,y

if __name__ == '__main__':
    train(X,y)