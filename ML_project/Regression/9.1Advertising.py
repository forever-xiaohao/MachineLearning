#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: haoc
@contact: 1012958103@qq.com
@software: PyCharm
@file: 9.1Advertising.py
@time: 2018/10/28 16:44
"""


import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
	path = '../dataset/Advertising.csv'
	# # 手写读取数据
	# f = file(path)
	# x = []
	# y = []
	# # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
	# for i, d in enumerate(f):
	# 	if i == 0: # 过滤掉第一行的值
	# 		continue
	# 	# Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
	# 	# 注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
	# 	d = d.strip()
	# 	if not d:
	# 		continue
	# 	# 定义：map(function, iterable, ...) 会根据提供的函数对指定序列做映射。
	# 	# 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
	# 	d = map(float, d.split(','))
	# 	x.append(d[1:-1])
	# 	y.append(d[-1])
	# print 'x的值：\n', x
	# print 'y的值：\n', y
	# x = np.array(x)
	# y = np.array(y)

	# # 使用Python自带库
	# f = file(path, 'r')
	# print f
	# d = csv.reader(f)
	# for line in d:
	# 	print line
	# f.close()

	# numpy读入
	p = np.loadtxt(path, delimiter=',',skiprows=1)
	print p
	print '\n\n==========\n\n'

	# pandas 读入
	data = pd.read_csv(path)    # TV、Radio、Newspaper、sales
	x = data[['TV', 'Radio', 'Newspaper']]
	y = data['Sales']
	print 'x的值：\n', x
	print 'y的值：\n', y

	# 解决绘图时中文乱码问题
	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False

	# 绘制1
	plt.figure(facecolor='w')    # 创建一个画板，并设置背景颜色为白色
	plt.plot(data['TV'], y, 'ro', label='TV')
	plt.plot(data['Radio'], y, 'g^', label='Radio')
	plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
	plt.legend(loc='lower right')	# 设置图标位置
	plt.xlabel(u'广告花费', fontsize=16)
	plt.ylabel(u'销售额', fontsize=16)
	plt.title(u'广告花费与销售额对比数据', fontsize=20)
	plt.grid()	# 图表带网格
	plt.show()

	# 绘制2
	plt.figure(facecolor='w', figsize=(9, 10))
	plt.subplot(311)	# 子图 311的意思：3行1列的第一个
	plt.plot(data['TV'], y, 'ro')
	plt.title('TV')
	plt.grid()
	plt.subplot(312)	# 312：3行1列的第2个
	plt.plot(data['Radio'], y, 'g^')
	plt.title('Radio')
	plt.grid()
	plt.subplot(313)	# 313: 3行1列第3个
	plt.plot(data['Newspaper'], y, 'b*')
	plt.title('Newspaper')
	plt.grid()
	plt.tight_layout()	# 设置距离边缘的距离
	plt.show()

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
	print type(x_test)
	print x_train.shape, y_train.shape
	linreg = LinearRegression()
	model = linreg.fit(x_train, y_train)
	print 'model:\n',model
	print linreg.coef_, linreg.intercept_    # 这两个参数分别是线性回归里的参数

	order = y_test.argsort(axis=0)	# 将数据集按大小排序,拿到对应的索引值
	y_test = y_test.values[order]	# 按照索引值对测试数据集取值
	x_test = x_test.values[order, :]
	y_hat = linreg.predict(x_test)	# 得到预测结果
	mse = np.average((y_hat - np.array(y_test)) ** 2)	# Mean Squared Error
	rmse = np.sqrt(mse)	# Root Mean Squared Error
	print 'MSE = ', mse
	print 'RMSE = ', rmse
	print 'R2 = ', linreg.score(x_train, y_train)	# 决定系数或者拟合程度
	print 'R2 = ', linreg.score(x_test, y_test)

	plt.figure(facecolor='w')
	t = np.arange(len(x_test))
	plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
	plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
	plt.legend(loc='upper right')
	plt.title(u'线性回归预测销量', fontsize=18)
	plt.grid(b=True)
	plt.show()





