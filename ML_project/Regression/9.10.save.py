#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: haoc
@contact: 1012958103@qq.com
@software: PyCharm
@file: 9.10.save.py
@time: 2018/10/30 23:11
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib

if __name__ == "__main__":
	data = pd.read_csv('../dataset/iris.data', header=None)
	x = data[[0, 1]]	 # 取数据集的前两列
	y = pd.Categorical(data[4]).codes	 # 将y值用0，1，2的形式表示

	if os.path.exists('iris.model'):
		print 'Load Model...'
		lr = joblib.load('iris.model')
	else:
		print 'Train Model...'
		lr = Pipeline([
			('sc', StandardScaler()),
			('poly', PolynomialFeatures(degree=3)),
			('clf', LogisticRegression())
		])
		lr.fit(x, y.ravel())	 # 训练模型
		y_hat = lr.predict(x)	 # 预测结果
		joblib.dump(lr, 'iris.model')	 # 将模型保存在本地,文件名为：iris.model
		print 'y_hat = \n', y_hat
		print 'accuracy= %.3f%%' % (100*accuracy_score(y, y_hat))
