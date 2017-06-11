# -*- coding: utf-8 -*-
"""
@Time : 2017/6/11 - 17:57
@Auther : Hao Chen
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pprint import pprint
from pandas_tool import inspect_dataset



if __name__ == "__main__":
    path = '../dataset/Advertising.csv'
    # pandas读入
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    # 查看数据的基本信息
    inspect_dataset(data)
    x = data[['TV', 'Radio']]    # 获取‘TV’和‘Radio’两列数据
    y = data['Sales']    # 获取‘Sales’数据
    print x
    print y

    # 解决绘图输出时的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 绘制1
    plt.figure(facecolor='w')    # 设置背景颜色为白色
    # 画出散点图
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
    plt.legend(loc='lower right')    # 设置图例的位置
    plt.xlabel(u'广告花费', fontsize=16)
    plt.ylabel(u'销售额', fontsize=16)
    plt.title(u'广告花费与销售额对比数据', fontsize=20)
    plt.grid()
    plt.show()

    # 绘制2
    plt.figure(facecolor='w', figsize=(9, 10))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()    # 设置图像距离画布的边界位置，这里采用默认值
    plt.show()

    # 划分训练集和测试集以及训练标签和测试标签
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    print type(x_test)
    print x_train.shape, y_train.shape
    linreg = LinearRegression()    # 构建线性回归模型
    model = linreg.fit(x_train, y_train)    # 训练该模型
    print model
    print '线性模型的系数：', linreg.coef_    # 得到线性模型的系数
    print '线性模型的独立性：', linreg.intercept_

    order = y_test.argsort(axis=0)    # 将测试标签排序
    y_test = y_test.values[order]     # 按排序后的索引位置获取测试标签
    x_test = x_test.values[order, :]  # 按排序后的索引位置获取测试集
    y_hat = linreg.predict(x_test)    # 根据测试集预测结果
    mse = np.average((y_hat - np.array(y_test)) ** 2)    # Mean Squared Error（均方误差）
    rmse = np.sqrt(mse)    # Root Mean Squared Error
    print 'MSE = ', mse,
    print 'RMSE = ', rmse
    print 'R2 = ', linreg.score(x_train, y_train)    # 得到训练数据的R2
    print 'R2 = ', linreg.score(x_test, y_test)      # 得到测试数据的R2

    plt.figure(facecolor='w')
    t = np.arange(len(x_test))     # 按照测试数据的长度创建等差数列
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.grid(b=True)
    plt.show()
