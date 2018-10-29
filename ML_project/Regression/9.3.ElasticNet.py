#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: haoc
@contact: 1012958103@qq.com
@software: PyCharm
@file: 9.3.ElasticNet.py
@time: 2018/10/29 23:14
"""
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

if __name__ == "__main__":
	warnings.filterwarnings(action='ignore', category=ConvergenceWarning)	 # 过滤掉警告信息
	np.random.seed(0)	 # 设置随机种子
	np.set_printoptions(linewidth=300)		# 设置输出的宽度
	N = 9
	x = np.linspace(0, 6, N) + np.random.randn(N)
	x = np.sort(x)
	y = x**2 - 4*x - 3 + np.random.randn(N)
	# 将x,y的值转化为列向量
	x.shape = -1, 1
	y.shape = -1, 1
	# 构造四个模型---Pipeline
	models = [
		Pipeline([
			('poly', PolynomialFeatures()),
			('linear', LinearRegression(fit_intercept=False))
		]),
		Pipeline([
			('poly', PolynomialFeatures()),
			('linear', RidgeCV(alphas=np.logspace(-3, 2, 10), fit_intercept=False))
		]),
		Pipeline([
			('poly', PolynomialFeatures()),
			('linear', LassoCV(alphas=np.logspace(-3, 2, 10), fit_intercept=False))
		]),
		Pipeline([
			('poly', PolynomialFeatures()),
			('linear', ElasticNetCV(alphas=np.logspace(-3, 2, 10), l1_ratio=[.1, .5, .7, .9, .95, .99, 1], fit_intercept=False))
		])
	]

	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False
	np.set_printoptions(suppress=True)

	plt.figure(figsize=(18, 12), facecolor='w')
	d_pool = np.arange(1, N, 1)    # 阶
	m = d_pool.size
	clrs = []    # 颜色
	for c in np.linspace(16711680, 255, m):
		clrs.append('#%06x' % c)
	line_width = np.linspace(5, 2, m)
	titles = u'线性回归', u'Ridge回归', u'LASSO', u'ElasticNet'
	tss_list = []
	rss_list = []
	ess_list = []
	ess_rss_list = []
	for t in range(4):
		model = models[t]
		plt.subplot(2, 2, t + 1)
		plt.plot(x, y, 'ro', ms=10, zorder=N)
		for i, d in enumerate(d_pool):
			model.set_params(poly__degree=d)
			model.fit(x, y.ravel())
			lin = model.get_params('linear')['linear']
			output = u'%s：%d阶，系数为：' % (titles[t], d)
			if hasattr(lin, 'alpha_'):
				idx = output.find(u'系数')
				output = output[:idx] + (u'alpha=%.6f，' % lin.alpha_) + output[idx:]
			if hasattr(lin, 'l1_ratio_'):  # 根据交叉验证结果，从输入l1_ratio(list)中选择的最优l1_ratio_(float)
				idx = output.find(u'系数')
				output = output[:idx] + (u'l1_ratio=%.6f，' % lin.l1_ratio_) + output[idx:]
			print output, lin.coef_.ravel()
			x_hat = np.linspace(x.min(), x.max(), num=100)
			x_hat.shape = -1, 1
			y_hat = model.predict(x_hat)
			s = model.score(x, y)
			r2, corr_coef = xss(y, model.predict(x))
			# print 'R2和相关系数：', r2, corr_coef
			# print 'R2：', s, '\n'
			z = N - 1 if (d == 2) else 0
			label = u'%d阶，$R^2$=%.3f' % (d, s)
			if hasattr(lin, 'l1_ratio_'):
				label += u'，L1 ratio=%.2f' % lin.l1_ratio_
			plt.plot(x_hat, y_hat, color=clrs[i], lw=line_width[i], alpha=0.75, label=label, zorder=z)
		plt.legend(loc='upper left')
		plt.grid(True)
		plt.title(titles[t], fontsize=18)
		plt.xlabel('X', fontsize=16)
		plt.ylabel('Y', fontsize=16)
	plt.tight_layout(1, rect=(0, 0, 1, 0.95))
	plt.suptitle(u'多项式曲线拟合比较', fontsize=22)
	plt.show()

	y_max = max(max(tss_list), max(ess_rss_list)) * 1.05
	plt.figure(figsize=(9, 7), facecolor='w')
	t = np.arange(len(tss_list))
	plt.plot(t, tss_list, 'ro-', lw=2, label=u'TSS(Total Sum of Squares)')
	plt.plot(t, ess_list, 'mo-', lw=1, label=u'ESS(Explained Sum of Squares)')
	plt.plot(t, rss_list, 'bo-', lw=1, label=u'RSS(Residual Sum of Squares)')
	plt.plot(t, ess_rss_list, 'go-', lw=2, label=u'ESS+RSS')
	plt.ylim((0, y_max))
	plt.legend(loc='center right')
	plt.xlabel(u'实验：线性回归/Ridge/LASSO/Elastic Net', fontsize=15)
	plt.ylabel(u'XSS值', fontsize=15)
	plt.title(u'总平方和TSS=？', fontsize=18)
	plt.grid(True)
	plt.show()