# -*- coding: utf-8 -*-
"""
@Time : 2017/6/3 - 15:02
@Auther : Hao Chen
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == "__main__":
    pd.set_option('display.width', 300)

    data = pd.read_csv('../dataset/tel.csv', skipinitialspace=True, thousands=',')    # thousands : str, default None 千分位分割符，如“，”或者“."
    print u'原始数据：\n', data.head(10)

    # print 'data.columns() = \n', data.columns

    # 将每列数据按照类别做Label，比如Married和Unmarried这两个值分别用0和1取代
    le = LabelEncoder()     # 编码标签值介于0和n 比如有5类则标签为0/1/2/3/4
    for col in data.columns:
        data[col] = le.fit_transform(data[col])    # 符合标签编码的返回编码标签

    print u'处理后数据1：\n', data.head(10)

    # 年龄分组
    # 将age这列的数据按照给定的bins半开区间做标记，比如年龄在[-1,6)标记为0；[6,12)标记为1；[12,18)标记为2 ；这里标记可以自己指定，但要和bins的取值个数一样
    bins = [-1, 6, 12, 18, 24, 35, 50, 70]
    data['age'] = pd.cut(data['age'], bins=bins, labels=np.arange(len(bins)-1))    # cut函数：返回指数每个x的值所属半开的范围，并且用labels的值标记
    # print u'处理后2：\n', data['age']

    # 取对数
    columns_log = ['income', 'tollten', 'longmon', 'tollmon', 'equipmon', 'cardmon',
                   'wiremon', 'longten', 'tollten', 'equipten', 'cardten', 'wireten', ]
    mms = MinMaxScaler()    # 这个估计量尺度和单独翻译每个特性,使其在训练集在给定的范围内,即在0和1之间。
    for col in columns_log:
        data[col] = np.log(data[col] - data[col].min() + 1)    # 求对数
        data[col] = mms.fit_transform(data[col].values.reshape(-1, 1))
    print u'处理后数据2：\n', data.head(10)

    # one-hot 编码
    columns_one_hot = ['region', 'age', 'address', 'ed', 'reside', 'custcat']
    for col in columns_one_hot:
        # get_dummies()函数：类别变量转换成虚拟变量/指标
        # join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。
        data = data.join(pd.get_dummies(data[col], prefix=col))
    print u'处理后数据3：\n', data.head(10)

    # 这里删除上面one-hot编码之后的列
    data.drop(columns_one_hot, axis=1, inplace=True)    # drop()函数：将指定的列按指定的方向删除，并返回

    print u'处理后数据4：\n', data.head(10)

    columns = list(data.columns)
    columns.remove('churn')
    x = data[columns]    # 得到数据
    y = data['churn']    # 得到类标记label
    print u'分组与one-hot编码后：\n', x.head(10)

    # 数组或矩阵分割成随机训练和测试子集
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=0)

    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=12, min_samples_split=5,
                                 oob_score=True, class_weight={0: 1, 1: 1/y_train.mean()})
    clf.fit(x_train, y_train)   # Build a forest of trees from the training set (X, y).

    # 特征选择   clf.feature_importances_ : 返回重要的特征权值或者说重要性
    important_features = pd.DataFrame(data={'features': x.columns, 'importance': clf.feature_importances_})
    print 'important_features:\n', important_features
    # 按照importance进行排序
    important_features.sort_values(by='importance', axis=0, ascending=False, inplace=True)
    important_features['cum_importance'] = important_features['importance'].cumsum()    # 返回累加的和
    print u'特征重要度：\n', important_features
    # 返回important_features['cum_importance']小于0.95的数据，取出‘features’这一列
    selected_features = important_features.loc[important_features['cum_importance'] < 0.95, 'features']

    # 重新组织数据
    x_train = x_train[selected_features]
    x_test = x_test[selected_features]

    # 模型训练
    clf.fit(x_train, y_train)
    print 'OOB Score: ', clf.oob_score_    # 训练数据集使用out-of-bag估计获得的分数
    y_train_pred = clf.predict(x_train)

    print u'训练集准确率：', accuracy_score(y_train.values, y_train_pred)
    print u'训练集查准率：', precision_score(y_train, y_train_pred)
    print u'训练集查全率：', recall_score(y_train, y_train_pred)
    print u'训练集f1 Score：', f1_score(y_train, y_train_pred)

    y_test_pred = clf.predict(x_test)
    print u'训练集准确率：', accuracy_score(y_test, y_test_pred)
    print u'训练集查准率：', precision_score(y_test, y_test_pred)
    print u'训练集查全率：', recall_score(y_test, y_test_pred)
    print u'训练集f1 Score：', f1_score(y_test, y_test_pred)


