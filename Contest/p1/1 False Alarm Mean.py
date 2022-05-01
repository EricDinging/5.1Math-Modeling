import numpy as np
import pandas as pd

# 读取数据
df1 = pd.read_csv('Attachment1.csv')
# 去掉极端值
df1_drop = df1[df1['误报次数(Number of false alarms)'] < 1000]
# 换算成“误报警率=误报警次数/一百万小时”
df1_drop = df1_drop.copy()
df1_drop['标准化误报次数'] = df1_drop['误报次数(Number of false alarms)'] / 432 * 1000000
# 计算不同部件各自均值
df = df1_drop.groupby('部件名称 (Component name)').aggregate(np.mean)
# 写入文件
df.to_csv('False Alarm Mean.csv')

