#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: haoc
@contact: 1012958103@qq.com
@software: PyCharm
@file: DecisionTreeRegressor.py
@time: 2018/12/18 13:19
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
import matplotlib.pyplot as plt


def create_data(n):
    np.random.seed(0)
    X = 5 * np.random.rand(n, 1)
    Y = np.sin(X).ravel()
    noise_num = (int)(n / 5)
    Y[::5] += 3 * (0.5 - np.random.rand(noise_num))
    return model_selection.train_test_split(X, Y, test_size=0.25, random_state=1)


def test_DecisionTreeRegressor(*data):
    X_train, X_test, Y_train, Y_test = data
    regr = DecisionTreeRegressor()
    regr.fit(X_train, Y_train)
    print("Training score:%f" % (regr.score(X_train, Y_train)))
    print("Testing score:%f" % (regr.score(X_test, Y_test)))
    #   绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y = regr.predict(x)
    ax.scatter(X_train, Y_train, label="train sample", c='g')
    ax.scatter(X_test, Y_test, label="test sample", c='r')
    ax.plot(x, y, label="predict_value", linewidth=2, alpha=0.5)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()


def test_DecisionTreeRegressor_splitter(*data):
    """
        检验随即划分与最优划分的影响;
        可以看到对于本问题，最优划分预测性能较强，但是相差不大。而对于训练集的拟合，二者都拟合的相当好。
    """
    X_train, X_test, Y_train, Y_test = data
    splitters = ['best', 'random']
    for splitter in splitters:
        regr = DecisionTreeRegressor(splitter=splitter)
        regr.fit(X_train, Y_train)
        print("Splitter %s" % splitter)
        print("Training score: %f" % (regr.score(X_train, Y_train)))
        print("Testing score: %f" % (regr.score(X_test, Y_test)))


def test_DecisionTreeRegressor_depth(*data):
    """
    考察决策树深度的影响。决策树的深度对应着树的复杂度。决策树越深，则模型越复杂
    :param data:
    :return:
    """
    X_train, X_test, Y_train, Y_test, maxdepth = data
    depths = np.arange(1, maxdepth)
    training_score = []
    testing_score = []
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth, splitter='best')
        regr.fit(X_train, Y_train)
        training_score.append(regr.score(X_train, Y_train))
        testing_score.append(regr.score(X_test, Y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_score, label="training score")
    ax.plot(depths, testing_score, label="testing score")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()





if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = create_data(100)
    # test_DecisionTreeRegressor(X_train, X_test, Y_train, Y_test)
    # test_DecisionTreeRegressor_splitter(X_train, X_test, Y_train, Y_test)
    test_DecisionTreeRegressor_depth(X_train, X_test, Y_train, Y_test, 20)
    pass
