#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: haoc
@contact: 1012958103@qq.com
@software: PyCharm
@file: 9.2.LinearRegression_CV.py
@time: 2018/10/29 17:14
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
	# pandas 读入
	data = pd.read_csv('../dataset/Advertising.csv')
	print data.head(10)
	# x = data[['TV', 'Radio', 'Newspaper']]
	x = data[['TV', 'Radio']]
	y = data['Sales']
	print x
	print y
	# 训练数据集和测试数据集分割，指定数据集选择的随机种子和训练集的占比
	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
	# model = Lasso()
	# Ridge 模型
	model = Ridge()
	alpha_can = np.logspace(-3, 2, 10)    # 10的-3次方到10的2次方，取十个数，即超参数集合
	np.set_printoptions(suppress=True)
	print 'alpha_can = ', alpha_can
	# 所有的参数对一一做验证，选择最优的参数，得到模型
	lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)    # 使用5折的Ridge做交叉回归
	lasso_model.fit(x_train, y_train)
	print '超参数：\n', lasso_model.best_params_

	order = y_test.argsort(axis=0)    # 得到排序之后的下标，默认是从小到大排序
	y_test = y_test.values[order]
	x_test = x_test.values[order, :]
	y_hat = lasso_model.predict(x_test)
	print 'R2: \n', lasso_model.score(x_test, y_test)
	mse = np.average((y_hat - np.array(y_test)) ** 2)    # Mean Squared Error
	rmse = np.sqrt(mse)    # Root Mean Squared Error
	print mse, rmse

	# ----------画图-----------
	t = np.arange(len(x_test))
	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False
	plt.figure(facecolor='w')    # 创建画板，并设置背景颜色为白色
	plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
	plt.plot(t, y_hat, 'g-', linewidth=2, label=u'测试数据')
	plt.title(u'线性回归预测销量', fontsize=18)
	plt.legend(loc='upper left')
	plt.grid(b=True, ls=':')
	plt.show()
