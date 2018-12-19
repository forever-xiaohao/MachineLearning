#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: haoc
@contact: 1012958103@qq.com
@software: PyCharm
@file: DecisionTreeClassifier方法解析.py
@time: 2018/12/18 19:51
"""
import numpy as np
import  matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection

"""
采用鸢尾花数据集，鸢尾花数据集一共有150个数据，这些数据分为3类（分别为setosa, versicolor, virginica）,
每类50个数据。每个数据包含4个属性：萼片长度、萼片宽度、花瓣长度、花瓣宽度。
"""


def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


def test_DecisionTreeClassifier(*data):
    X_train, X_test, y_train, y_test = data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print("Training score:%f" % (clf.score(X_train, y_train)))
    print("Testing score:%f" % (clf.score(X_test, y_test)))


def test_DecisionTreeClassifier_criterion(*data):
    X_train, X_test, y_train, y_test = data
    criterions = ['gini', 'entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train, y_train)
        print("criterion:%s" % criterion)
        print("Training score:%f" % (clf.score(X_train, y_train)))


def test_DecisionTreeClassifier_depth(*data):
    X_train, X_test, y_train, y_test, maxdepth=data
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label="traing score", marker='o')
    ax.plot(depths, testing_scores, label="testing score", marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5, loc='best')
    plt.show()


if __name__ == "__main__":
    # X_train, y_train, X_test, y_test = load_data()
    # print X_train
    # test_DecisionTreeClassifier(X_train, y_train, X_test, y_test)
    X_train, X_test, y_train, y_test = load_data()
    test_DecisionTreeClassifier_depth(X_train, X_test, y_train, y_test, 20)