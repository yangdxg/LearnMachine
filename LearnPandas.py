# 1. 下载学习pandas使用的数据集
# Pandas术语：数据列成为系列，表格称为数据库
import os
import pandas as pd
import requests

r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with open('iris.data','w') as f:
    f.write(r.text)
# 从csv文件导入数据,并添加标题行
df = pd.read_csv('iris.data',names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','类别'])
# df.head(n)查看DataFormate前几行
print(df.head(2))

# 2.通过列名，从数据框中选择某一列
print('--------------练习2---------------')
print(df['花萼长度'].head(2))
# 3.执行数据切片 .ix[row,column]
print('--------------练习3---------------')
# 选择前俩列，前四行
print(df.ix[:3,:2])
# 4. 使用列表迭代器并只选择描述width的列
print('--------------练习4---------------')
# df.columns会返回所有的列名
print(df.ix[:3,[x for x in df.columns if '宽度' in x]])
print('--------------练习5---------------')
# unique()去重
print(df['类别'].unique())
print('--------------练习6---------------')
# 选取类别为'Iris-virginica'的数据
print(df[df['类别']=='Iris-virginica'])
# 每个列下面的行数
print(df.count())
# 筛选类别为'Iris-virginica'下每个列下面的行数
print(df[df['类别']=='Iris-virginica'].count())
print('--------------练习7---------------')
# 左侧的索引保留了原始行号，将筛选后的数据保存为一个新的数据框并重置索引
virginica = df[df['类别']=='Iris-virginica'].reset_index(drop=True)
print(virginica)
print('--------------练习8---------------')
# 通过在某个列上使用多个条件来选择数据
# 选择类别为'Iris-virginica'并且花萼宽度大于2.2的数据
print(df[(df['类别']=='Iris-virginica')&(df['花萼宽度']>2.2)])
print('--------------练习8---------------')
# 获取各列的描述性统计信息，类别信息被自动删除，因为它在这里是不相关的
print(df.describe())
# 传入自定义的百分比，获取更为详细的信息
print(df.describe(percentiles=[.20,.40,.60,.80]))
print('--------------练习9---------------')
# 检查这些特征之间的相关性，可以通过数据框上调用'.corr()'来完成
# 返回每个行-列对中的Pearson相关系数
print(df.corr())
# 通过传递参数，切换到Kendall's tau或Spearman's秩相关系数
print('Kendalls tau 系数')
print(df.corr(method='spearman'))
# print('Spearmans 系数')
# print(df.corr(method='kendall'))
