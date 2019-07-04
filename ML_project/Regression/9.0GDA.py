#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: haoc
@contact: 1012958103@qq.com
@software: PyCharm
@file: 9.0GDA.py
@time: 2018/10/27 21:42
"""
import math

if __name__ == "__main__":
	learning_rate = 0.01
	for a in range(1, 100):
		cur = 0
		for i in range(1000):
			cur -= learning_rate*(cur**2 - a)
		print ' %d的平方根（近似）为：%.8f,真实值是：%.8f' %(a, cur, math.sqrt(a))
