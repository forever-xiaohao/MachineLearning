# -*- coding: utf-8 -*-
"""
@Time : 2017/6/2 - 19:59
@Auther : Hao Chen
"""
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def enum_row(row):
    print row['state']
def find_state_code(row):
    if row['state'] != 0:
        print process.extractOne(row['state'], states, score_cutoff=80)
def capital(str):
    return str.capitalize()


def correct_state(row):
    if row['state'] != 0:
        state = process.extractOne(row['state'], states, score_cutoff=80)
        if state:
            state_name = state[0]
            return ' '.join(map(capital, state_name.split(' ')))
    return row['state']


def fill_state_code(row):
    if row['state'] != 0:
        state = process.extractOne(row['state'], states, score_cutoff=80)
        if state:
            state_name = state[0]
            return state_to_code[state_name]
    return ''

if __name__ == "__main__":
    pd.set_option('display.width', 200)    # 设置显示的宽度
    data = pd.read_excel('../dataset/sales.xlsx', sheetname='sheet1', header=0)      #  读取Excel表格，sheetname:表示读取的表格名，header=0表示不读取第一行
    print 'data.head() = \n', data.head()
    print 'data.tail() = \n', data.tail()
    print 'data.dtypes = \n', data.dtypes
    print 'data.columns = \n', data.columns
    for c in data.columns:
        print c,
    print
    data['total'] = data['Jan'] + data['Feb'] + data['Mar']
    print data.head()
    print data['Jan'].sum()
    print data['Jan'].min()
    print data['Jan'].max()
    print data['Jan'].mean()
    print "这是测试"

    print '=========================='
    # 添加一行
    s1 = data[['Jan', 'Feb', 'Mar', 'total']].sum()
    print s1
    s2 = pd.DataFrame(data=s1)    # 创建DataFrame格式的s2,其值为s1
    print 's2 = \n', s2
    print 's2.T = \n', s2.T    # 输出s2的转置
    print s2.T.reindex(columns=data.columns)    # 设置s2转置的索引
    #
    s = pd.DataFrame(data=data[['Jan', 'Feb', 'Mar', 'total']].sum()).T
    s = s.reindex(columns=data.columns, fill_value=0)    # 参数fill_value表示没有值得填充为0
    print s
    data = data.append(s,ignore_index=True)    # 将结果追加到原数据集末尾，并忽略索引
    data = data.rename(index={15: 'Total'})    # 设置第15行的行索引为“Total”
    print data.tail()

    # apply的使用
    print '=====================apply的使用========================='
    print 'data.apply() = \n', data.apply(enum_row, axis=1)

    state_to_code = {"VERMONT": "VT", "GEORGIA": "GA", "IOWA": "IA", "Armed Forces Pacific": "AP", "GUAM": "GU",
                     "KANSAS": "KS", "FLORIDA": "FL", "AMERICAN SAMOA": "AS", "NORTH CAROLINA": "NC", "HAWAII": "HI",
                     "NEW YORK": "NY", "CALIFORNIA": "CA", "ALABAMA": "AL", "IDAHO": "ID",
                     "FEDERATED STATES OF MICRONESIA": "FM",
                     "Armed Forces Americas": "AA", "DELAWARE": "DE", "ALASKA": "AK", "ILLINOIS": "IL",
                     "Armed Forces Africa": "AE", "SOUTH DAKOTA": "SD", "CONNECTICUT": "CT", "MONTANA": "MT",
                     "MASSACHUSETTS": "MA",
                     "PUERTO RICO": "PR", "Armed Forces Canada": "AE", "NEW HAMPSHIRE": "NH", "MARYLAND": "MD",
                     "NEW MEXICO": "NM",
                     "MISSISSIPPI": "MS", "TENNESSEE": "TN", "PALAU": "PW", "COLORADO": "CO",
                     "Armed Forces Middle East": "AE",
                     "NEW JERSEY": "NJ", "UTAH": "UT", "MICHIGAN": "MI", "WEST VIRGINIA": "WV", "WASHINGTON": "WA",
                     "MINNESOTA": "MN", "OREGON": "OR", "VIRGINIA": "VA", "VIRGIN ISLANDS": "VI",
                     "MARSHALL ISLANDS": "MH",
                     "WYOMING": "WY", "OHIO": "OH", "SOUTH CAROLINA": "SC", "INDIANA": "IN", "NEVADA": "NV",
                     "LOUISIANA": "LA",
                     "NORTHERN MARIANA ISLANDS": "MP", "NEBRASKA": "NE", "ARIZONA": "AZ", "WISCONSIN": "WI",
                     "NORTH DAKOTA": "ND",
                     "Armed Forces Europe": "AE", "PENNSYLVANIA": "PA", "OKLAHOMA": "OK", "KENTUCKY": "KY",
                     "RHODE ISLAND": "RI",
                     "DISTRICT OF COLUMBIA": "DC", "ARKANSAS": "AR", "MISSOURI": "MO", "TEXAS": "TX", "MAINE": "ME"}
    states = state_to_code.keys()    # 得到字典的key
    print fuzz.ratio('Python Package', 'PythonPackage')
    print process.extract('Mississippi', states)
    print process.extract('Mississipi', states, limit=1)
    print process.extractOne('Mississipi', states)
    data.apply(find_state_code, axis=1)

    print 'Before Correct State:\n', data['state']
    data['state'] = data.apply(correct_state, axis=1)
    print 'After Correct State:\n', data['state']
    data.insert(5, 'State Code', np.nan)
    data['State Code'] = data.apply(fill_state_code, axis=1)
    print data

    # group by
    print '========================group by======================'
    print data.groupby('State Code')
    print 'All Columns:\n'
    print data.groupby('State Code').sum()
    print 'Short Columns:\n'
    print data[['State Code', 'Jan', 'Feb', 'Mar', 'total']].groupby('State Code').sum()

    # 写入文件
    data.to_excel('../dataout/sales_result.xls', sheet_name='Sheet1', index=False)
