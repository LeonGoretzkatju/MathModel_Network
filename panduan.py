# coding:utf-8
import pandas as pd

data = pd.read_excel('因子数据.xlsx', sheet_name='Sheet1',usecols=[2])
for i in range(5,32):
    if data[i]>=50 or data[i]<=120:
        data[i] = 7
    else:
        data[i] = 10

print(data)