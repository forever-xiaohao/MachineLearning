# -*- coding: utf-8 -*-
"""
@Time : 2017/6/2 - 18:35
@Auther : Hao Chen
道路拥堵模型
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def clip(x,path):
    for i in range(len(x)):
        if x[i] >= path:
            x[i] %= path
if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = 5000    # 环形公路的长度
    n = 100        # 公路中的车辆数目
    v0 = 50        # 车辆的初始速度
    p = 0.3        # 随机减速概率
    Times = 3000

    np.random.seed(0)     # 设置随机种子
    x = np.random.rand(n) * path
    x.sort()
    v = np.tile([v0], n).astype(np.float)    # tile函数功能：重复某个数组，这里是重复[v0],n次

    plt.figure(figsize=(10, 8), facecolor='w')    # 设置画板的大小（10,8）表示英寸
    for t in range(Times):
        plt.scatter(x, [t]*n, s=1, c='k', alpha=0.05)    # 绘制散点图，s 表示点点的大小，c表示颜色，alpha表示点的亮度
        for i in range(n):
            if x[(i+1) % n] > [i]:
                d = x[(i+1) % n] - x[i]    # 距离前车的距离
            else:
                d = path - x[i] + x[(i+1) % n]
            if v[i] < d:
                if np.random.rand() > p:
                    v[i] += 1
                else:
                    v[i] -= 1
            else:
                v[i] = d - 1
        v = v.clip(0, 150)    # clip()这个方法会给出一个区间，在区间之外的数字将被剪除到区间的边缘，例如给定一个区间[0,1]，则小于0的将变成0，大于1则变成1
        x += v
        clip(x, path)
    plt.xlim(0, path)     # 设置x轴的最小值和最大值
    plt.ylim(0, path)     # 设置y轴的最小值和最大值
    plt.xlabel(u'车辆位置', fontsize=16)
    plt.ylabel(u'模拟时间', fontsize=16)
    plt.title(u'环形公路车辆堵车模拟', fontsize=20)
    plt.tight_layout(pad=2)   # 自动调整次要情节参数给指定的填充。
    plt.show()
