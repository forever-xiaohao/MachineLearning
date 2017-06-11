# -*- coding: utf-8 -*-
"""
@Time : 2017/6/11 - 19:14
@Auther : Hao Chen
"""
import pandas  as pd

def inspect_dataset(df_data):
    """
    查看数据集基本信息
    :param df_data:
    :return:
    """
    print '数据集基本信息：\n'
    print df_data.info()

    print '数据集有%i行，%i列'%(df_data.shape[0], df_data.shape[1])
    print '数据预览：\n'
    print df_data.head(10)