# -*- coding: utf-8 -*-
"""
@Time : 2017/6/4 - 13:02
@Auther : Hao Chen
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,SelectPercentile,chi2
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def extend(a, b):
    return 1.05*a-0.05*b, 1.05*b-0.05*a

if __name__ == '__main__':
    stype = 'pca'
    pd.set_option('display.width', 200)
    data = pd.read_csv('../dataset/iris.data', header=None)
    # print '原始数据：\n', data.head(5)
    columns = np.array([u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度', u'类型'])
    # print zip(np.arange(5), columns)
    data.rename(columns=dict(zip(np.arange(5), columns)), inplace=True)
    print '修改后数据：\n', data.head(5)
    # 将label标记变为0,1,2
    data[u'类型'] = pd.Categorical(data[u'类型']).codes
    # print data[u'类型']
    print data.head(5)
    x = data[columns[:-1]]    # 取出第一列到最后一列的所有数据（不包括最后一列）
    # print 'x= \n', x
    y = data[columns[-1]]    # 取出最后一列
    # print 'y= \n', y

    if stype == 'pca':
        pca = PCA(n_components=2, whiten=True, random_state=0)
        x = pca.fit_transform(x)    # 返回降维后的数据
        print '降维后的数据x=\n', x
        print '各方向方差：', pca.explained_variance_    # 输出各个方向的方差
        print '方差所占比例：', pca.explained_variance_ratio_    # 返回 所保留的n个成分各自的方差百分比。
        x1_label, x2_label = u'组分1', u'组分2'
        title = u'鸢尾花数据PCA降维'
    else:
        # 选择K个最好的特征，返回选择特征后的数据
        # 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
        # 参数k为选择的特征个数
        fs = SelectKBest(chi2, k=2)
        fs.fit(x, y)
        idx = fs.get_support(indices=True)
        print 'fs.get_support() = \n', idx
        x = x[idx]
        x = x.values    # 为下面使用方便，DataFrame转换成ndarray
        x1_label, x2_label = columns[idx]
        title = u'鸢尾花数据特征选择'
    print x[:5]
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    # 绘制散点图
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, marker='o', cmap=cm_dark)
    plt.grid(b=True, ls=':')
    plt.xlabel(x1_label, fontsize=14)
    plt.ylabel(x2_label, fontsize=14)
    plt.title(title, fontsize=18)
    plt.show()


    # 分割训练集和测试集
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.7)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=True)),
        ('lr', LogisticRegressionCV(Cs=np.logspace(-3, 4, 8), cv=5, fit_intercept=False))
    ])
    model.fit(x, y)
    print '最优参数：', model.get_params('lr')['lr'].C_
    y_hat = model.predict(x)
    print '训练集精确度：', metrics.accuracy_score(y, y_hat)
    y_test_hat = model.predict(x_test)
    print '测试集精确度：', metrics.accuracy_score(y_test, y_test_hat)

    N, M = 500, 500 # 横纵各采样多少个值
    x1_min, x1_max = extend(x[:, 0].min(), x[:, 0].max())  # 第0列的范围
    x2_min, x2_max = extend(x[:, 1].min(), x[:, 1].max())  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)     # linspace() 返回等差数列
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    y_hat = model.predict(x_show)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, edgecolors='k', cmap=cm_dark)  # 样本的显示
    plt.xlabel(x1_label, fontsize=14)
    plt.ylabel(x2_label, fontsize=14)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(b=True, ls=':')
    # 画各种图
    # a = mpl.patches.Wedge(((x1_min+x1_max)/2, (x2_min+x2_max)/2), 1.5, 0, 360, width=0.5, alpha=0.5, color='r')
    # plt.gca().add_patch(a)
    patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
              mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
              mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]
    plt.legend(handles=patchs, fancybox=True, framealpha=0.8, loc='lower right')
    plt.title(u'鸢尾花Logistic回归分类效果', fontsize=17)
    plt.show()
