# -*- coding: utf-8 -*-
"""
@Time : 2017/6/12 - 20:51
@Auther : Hao Chen
模拟数据计算roc和auc
"""
import numbers
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize     # 做one-hot编码
from numpy import interp
from sklearn import metrics
from itertools import cycle



if __name__ == '__main__':
    np.random.seed(0)    # 设置随机数种子
    pd.set_option('display.width', 300)
    np.set_printoptions(suppress=True)
    n = 300
    x = np.random.randn(n, 50)    # 生成300*50的数据
    y = np.array([0]*100 + [1]*100 + [2]*100)    # 将数据标记为3类，每100个数据为一类
    n_class = 3    # 类别数

    # alpha = np.logspace(-3, 3, 7)
    # 逻辑回归   参数penalty：表示一个字符串，指定了正则化策略。
    #                          'l2'：表示优化目标函数为l2正则
    #                          'l1'：表示优化目标函数为l1正则
    #            参数C：表示一个浮点数。它指定了罚项系数的倒数。如果它的值越小，则正则化项越大
    clf = LogisticRegression(penalty='l2', C=1)
    clf.fit(x, y)    # 训练模型
    y_score = clf.decision_function(x)    # 因为这里有3类，可以用此函数做3分类的回归
    y = label_binarize(y, classes=np.arange(n_class))    # 将类别个数做one-hot编码
    color = cycle('gbc')    # 无穷循环器 #重复序列的元素
    fpr = dict()
    tpr = dict()
    auc = np.empty(n_class + 2)   # 生成一个随机数组，参数为指定的个数
    # 设置字体和编码格式防止乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # 创建画布
    plt.figure(figsize=(7, 6), facecolor='w')
    # zip()函数，返回一个元祖，第一个参数对应key，第二个参数对应value
    for i, color in zip(np.arange(n_class), color):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y[:, i], y_score[:, i])
        auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], c=color, lw=1.5, alpha=0.7, label=u'AUC=%.3f' % auc[i])
    # micro   ravel():将多维数据降为一维
    fpr['micro'], tpr['micro'], thresholds = metrics.roc_curve(y.ravel(), y_score.ravel())
    auc[n_class] = metrics.auc(fpr['micro'], tpr['micro'])
    plt.plot(fpr['micro'], tpr['micro'], c='r', lw=2, ls='-', alpha=0.8, label=u'micro, AUC=%.3f' % auc[n_class])
    # maroc  unique（）保留数组中不同的值，返回两个参数。
    fpr['macro'] = np.unique(np.concatenate([fpr[i] for i in np.arange(n_class)]))
    tpr_ = np.zeros_like(fpr['macro'])
    for i in np.arange(n_class):
        tpr_ += interp(fpr['macro'], fpr[i], tpr[i])
    tpr_ /= n_class
    tpr['macro'] = tpr_
    auc[n_class + 1] = metrics.auc(fpr['macro'], tpr['macro'])
    print auc
    print 'Macro AUC:', metrics.roc_auc_score(y, y_score, average='macro')
    plt.plot(fpr['macro'], tpr['macro'], c='m', lw=2, alpha=0.8, label=u'macro，AUC=%.3f' % auc[n_class + 1])
    plt.plot((0, 1), (0, 1), c='#808080', lw=1.5, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True)
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    # plt.legend(loc='lower right', fancybox=True, framealpha=0.8, edgecolor='#303030', fontsize=12)
    plt.title(u'ROC和AUC', fontsize=17)
    plt.show()