import numpy as np
import pandas as pd
df1 = pd.read_csv('Attachment1.csv')
df1.groupby('部件名称 (Component name)').aggregate(np.mean)
