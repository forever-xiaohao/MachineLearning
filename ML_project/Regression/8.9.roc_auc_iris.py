# -*- coding: utf-8 -*-
"""
@Time : 2017/6/13 - 12:26
@Auther : Hao Chen
"""
import numpy as np
import pandas as pd
from pandas_tool import inspect_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import cycle


if __name__ == '__main__':
    np.random.seed(0)    # 设置随机种子
    pd.set_option('display.width', 300)
    np.set_printoptions(suppress=True)
    # 读取数据
    data = pd.read_csv('../dataset/iris.data', header=None)
    # 查看数据
    inspect_dataset(data)
    iris_types = data[4].unique()
    for i, iris_type in enumerate(iris_types):
        data.set_value(data[4] == iris_type, 4, i)    # 第一个参数是原数值，第二个参数是key，第三个参数是value
    inspect_dataset(data)
    x = data.iloc[:, :2]    # 取出前两列数据
    n, features = x.shape
    print x
    y = data.iloc[:, -1].astype(np.int)    # 取出最后一列数据，并转换为int类型
    c_number = np.unique(y).size   # 得到类别的个数

    # 分割测试数据和训练数据
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.6, random_state=0)
    # 对y标签one-hot编码
    y_one_hot = label_binarize(y_test, classes=np.arange(c_number))
    print 'y_one_hot = \n', y_one_hot
    print 'y_one_hot.ravel() = \n', y_one_hot.ravel()
    # 设置超参数
    alpha = np.logspace(-2, 2, 20)
    models = [
        ['KNN', KNeighborsClassifier(n_neighbors=7)],
        ['LogisticRegression', LogisticRegressionCV(Cs=alpha, penalty='l2', cv=3)],
        ['SVM(Linear)', GridSearchCV(SVC(kernel='linear', decision_function_shape='ovr'), param_grid={'C': alpha})],
        ['SVM(RBF)', GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr'), param_grid={'C': alpha, 'gamma': alpha})]
    ]
    colors = cycle('gmcr')
    # 设置字体和编码格式，避免中文乱码
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # 创建画板
    plt.figure(figsize=(7,6), facecolor='w')
    for (name, model), color in zip(models, colors):
        model.fit(x, y)    # 训练模型
        if hasattr(model, 'C_'):    # 判断模型中是否包含'C_'参数
            print model
            print 'model.C_ = \n', model.C_
        if hasattr(model, 'best_params_'):
            print model
            print 'model.best_params_ =\n', model.best_params_
        if hasattr(model, 'predict_proba'):
            print model
            y_score = model.predict_proba(x_test)
            print ' y_score.ravel() = \n',  y_score.ravel()
        else:
            print model
            y_score = model.decision_function(x_test)
            print ' y_score.ravel() = \n',  y_score.ravel()
        # 评价分类效果--AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
        auc = metrics.auc(fpr, tpr)
        print 'auc = \n', auc
        plt.plot(fpr, tpr, c=color, lw=2, alpha=0.7, label=u'%s, AUC=%.3f' % (name, auc))
    plt.plot((0, 1), (0, 1), c='#808080', lw=2, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'鸢尾花数据不同分类器的ROC和AUC', fontsize=17)
    plt.show()







