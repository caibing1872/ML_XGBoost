import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split   # cross_validation
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    path = u'iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x = data.loc[:, range(4)]
    y = pd.Categorical(data[4]).codes
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}

    bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
    y_hat = bst.predict(data_test)
    result = y_test.reshape(1, -1) == y_hat
    print ('正确率:\t', float(np.sum(result)) / len(y_hat))
    print ('END.....\n')

    models = [('LogisticRegression' , LogisticRegressionCV(Cs = 10 , cv = 3)), #超参数 Cs
              ('RandomForest' , RandomForestClassifier(n_estimators = 30 , criterion = 'gini'))]
    for name , model in models:
        model.fit(x_train ,y_train)
        print (name , '训练集正确率： ' , accuracy_score(y_train , model.predict(x_train)))
        print (name, '测试集正确率： ', accuracy_score(y_test, model.predict(x_test)))

        #在意分类精度，维度不是很高 选择SVM
        #实际情况，不在意 泛化能力 ，选择LogisticRegression
        #样本不均衡 一般处理 先采样，后归一化