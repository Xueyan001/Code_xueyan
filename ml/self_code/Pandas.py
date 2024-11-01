import pandas as pd
import numpy as np

# 创建DataFrame
# data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [True,True,True]}
# df = pd.DataFrame(data)
# df['D'] = [True, False, True]
# filter_true = df[df['C'] & df['D']]

# print(filter_true)
#sort
# data = {'A': [4,1,9], 'B': [1,2,3]}
# df = pd.DataFrame(data)
# data_sorted = df.sort_values(by='A', ascending=True, inplace=False)
# print(data_sorted)

#classify
data = {
    'Category': ['A', 'B', 'A', 'A', 'B', 'C'],
    'Value': [10, 20, 30, 40, 50, 60]
}
df = pd.DataFrame(data)
grouped = df.groupby('Category')
sum_df = df.groupby('Category')['Value'].sum()
agg_df = df.groupby('Category')['Value'].agg(['sum', 'mean'])
print(agg_df)
#print(sum_df)


# 创建Series
# s = pd.Series([1.0, 2.0, 3.0, 4.0], index=['a', 'b', 'c', 'd'], name='MySeries')
# s = pd.Series(['s1', 's2', 's3', 's4'], index=['a', 'b', 'c', 'd'], name='MySeries')

# 基本属性
# print(df)
# print(df.at[2, 'B'])
#print(df.at[0, 'A'])# output(0,'A')element
# print(df.shape)  # 输出：(3, 2)
# print(s)
# print(s.index)  # 输出：Index(['a', 'b', 'c', 'd'], dtype='object')
# print(s.name)  # 输出：'MySeries'
