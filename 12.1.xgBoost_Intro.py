import xgboost as xgb
import numpy as np
# 1、xgBoost的基本使用
# 2、自定义损失函数的梯度和二阶导
# 3、binary:logistic/logitraw

# 定义f: theta * x
def g_h(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h


def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


if __name__ == "__main__":
    # 读取数据
    data_train = xgb.DMatrix('agaricus_train.txt')   #读取训练数据
    data_test = xgb.DMatrix('agaricus_test.txt')     #自动读取测试数据
    print ('data_train: ' , data_train)
    print ('type(data_train): ' , type(data_train))

    # 设置参数
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'} # logitraw
    # 'eta给定为1 ，表示要做若干个分类器，是为了防止过拟合，此处给1 表示的是不考虑';  'silent': 1 表示的是 输出信息少一点，为0，多输出一些；
    # param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')] #实时观测 训练的精确度
    n_round = 7  #需要做几棵树 确定第几次效果好
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=g_h, feval=error_rate)
    #data_train 是包含两部分数据，即包含 x与y数据 ; ‘obj=g_h’表示的是 目标函数 自定义一阶导和二阶导
    #######################
    # XGboost与决策树相比 ，数量少，速度快
    #######################

    # 计算错误率
    np.set_printoptions(suppress=True)
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print ('y_hat: ' , y_hat)
    print ('y: ' , y)
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print ('样本总数：\t', len(y_hat))
    print ('错误数目：\t%4d' % error)
    print ('错误率：\t%.5f%%' % (100*error_rate))
