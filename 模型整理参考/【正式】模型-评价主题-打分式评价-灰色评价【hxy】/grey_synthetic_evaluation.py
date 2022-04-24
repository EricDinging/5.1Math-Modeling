'''
    This program using Grey Synthetix Evaluation Method to make an evaluation.

    Input: 原始数据矩阵Y（第一行为参考序列，其他为比较序列；指标为列名称)
    Output: 灰色关联度矩阵A及最优结果
'''

import pandas as pd
import numpy as np

# 读取数据
data = pd.read_excel("/Users/xinyuanhe/Desktop/data.xlsx")

# 设置参考序列
reference = data[0:1]

