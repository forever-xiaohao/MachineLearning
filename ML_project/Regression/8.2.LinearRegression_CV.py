# -*- coding: utf-8 -*-
"""
@Time : 2017/6/12 - 18:32
@Auther : Hao Chen
线性回归加入调参的功能
"""
import numpy as np
import pandas as pd
from pandas_tool import inspect_dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    # pandas读入数据
    data = pd.read_csv('../dataset/Advertising.csv')
    # 查看数据的基本信息
    inspect_dataset(data)
    # 得到指定列的数据
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    print x
    print y

    # 分割数据得到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # 调用模型
    model = Lasso()
    # model = Ridge()
    # 设置超参数，这里是自己定义，一个等比数列数组
    alpha_can = np.logspace(-3, 2, 10)
    np.set_printoptions(suppress=True)    # 设置输出格式为小数点形式
    print 'alpha_can = ', alpha_can

    # 通过交叉验证选择最优参数,参数cv=5表示交叉次数为5
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    # 在训练集上训练数据
    lasso_model.fit(x_train, y_train)
    # 输出最优参数
    print '超参数：\n', lasso_model.best_params_

    # 通过测试集进行验证
    order = y_test.argsort(axis=0)    # 将测试标签按照从小到大顺序排序
    y_test = y_test.values[order]     # 按照排序好的索引位置取标签
    x_test = x_test.values[order, :]    # 按照排序好的索引位置取对应的测试数据
    # 得到预测结果
    y_hat = lasso_model.predict(x_test)
    # 得到在该模型上的测试结果
    print lasso_model.score(x_test, y_test)
    # 计算均方误差
    mse = np.average((y_hat - np.array(y_test)) ** 2)
    rmse = np.sqrt(mse)    # Root Mean Squared Error
    print mse, rmse

    t = np.arange(len(x_test))    # 得到和训练数据集长度的数据
    # 设置字体和编码方式，避免画图时出现中文乱码
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')    # 创建一个画板，设置背景颜色为白色
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.legend(loc='upper right')    # 设置用例图标的位置
    plt.tight_layout()
    plt.grid()    # 画出网格
    plt.show()

