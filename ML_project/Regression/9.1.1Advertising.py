#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: ThinkPad
@contact: 1012958103@qq.com
@software: PyCharm
@file: 9.1.1Advertising.py
@time: 2019/7/4 23:18
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    path = '../dataset/Advertising.csv'

    # numpy 读取
    p = np.loadtxt(path, delimiter=',', skiprows=1)
    print p
    print '\n\n==============\n\n'

    # pandas 读取
    data = pd.read_csv(path)  # TV、Radio、Newspaper、sales
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    print 'X的值：\n', x
    print 'Y的值：\n', y

    # 解决绘图时中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 绘制图1
    plt.figure(facecolor='w')    # 创建一个画板，并设置背景颜色为白色
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
    plt.legend(loc='lower right')       # 指定图标位置
    plt.xlabel(u'广告话费', fontsize=16)
    plt.ylabel(u'销售额', fontsize=16)
    plt.title(u'广告花费与销售额对比数据', fontsize=20)
    plt.grid()    # 设置图表带表格
    plt.show()

    # 绘制图2
    plt.figure(facecolor='w', figsize=(9, 10))
    plt.subplot(311)    # 子图 311 的意思：3行1列的第一个
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)    # 312: 3行1列的第2个
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)    # 313: 3行1列第3个
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()    # 设置距离边缘的距离
    plt.show()

    '''
    使用线性回归方法训练数据
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    print type(x_test)
    print x_train.shape, y_train.shape
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)     # 训练模型
    print 'model:\n', model
    print linreg.coef_, linreg.intercept_



