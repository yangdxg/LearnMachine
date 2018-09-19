# 学习操作及处理数据
import pandas as pd
import numpy as np

df = pd.read_csv('Iris.data', names=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类别'])
print('--------------练习1(Map)---------------')
# Map方法适用于序列数据
# 使用Map方法经一个Python字典作为参数
df['类别'] = df['类别'].map({'Iris-setosa': 'SET', 'Iris-versicolor': 'VER', 'Iris-virginica': 'VIR'})
print(df)
print('--------------练习2(Apply)---------------')
# Apply方法既可以在数据库上工作，也可以在序列上工作
df['宽花瓣'] = df['花瓣宽度'].apply(lambda v: 1 if v >= 1.3 else 0)  # 花瓣宽度大于1.3，宽花瓣值为1，否则为0
print(df)
df['花瓣面积'] = df.apply(lambda r: r['花瓣长度'] * r['花瓣宽度'], axis=1)  # axis=1代表pandas对行操作，axis=0代表pandas对列操作
print(df)
print('--------------练习3(Applymap)---------------')
#对数据框里所有的数据单元执行一个函数，使用applymap
#如果某个值是float类型的实例，使用numpy库返回np.log返回该值(对数)
df1=df.applymap(lambda v: np.log(v) if isinstance(v, float) else v)
print(df1)
print('--------------练习4(Groupby)---------------')
# 基于某些你所选择的类别对数据进行分组
# 重新导入iris数据集
print(df.groupby('类别').mean())
# 获取描述性统计信息
print(df.groupby('类别').describe())
print(df.groupby('花瓣宽度')['类别'].unique().to_frame())
# 自定义的聚集函数
print(df.groupby('类别')['花瓣宽度'].agg({'间距':lambda x:x.max() - x.min(),'最大值':np.max,'max':np.max,'最小值':np.min}))