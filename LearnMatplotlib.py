# Matplotlib库的使用(绘图库)
import pandas as pd
# 引入matplotlib
import matplotlib.pyplot as plt
# 设置风格为近似R中的ggplot库
plt.style.use('ggplot')
# 设置在记事本中可见
# %matplotlib inline(pycharm中没有效果，在pycharm中最后使用plt.show()展示)
# 引入bumpy
import numpy as np
# 重新对取花瓣数据
df = pd.read_csv('iris.data',names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','类别'])
print('--------------练习1---------------')
# 解决显示中文问题
from matplotlib.font_manager import FontManager, FontProperties
font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

fig,ax = plt.subplots(figsize=(6,4))#创建了宽为6英寸高为4英寸的一个插图
ax.hist(df['花瓣宽度'],color='black');# 调用.hist()并传入数据，依照iris数据框绘制了花瓣宽度直方图
ax.set_ylabel('数量',fontsize=12,fontproperties=font)
ax.set_xlabel('宽度',fontsize=12,fontproperties=font)
plt.title('花瓣宽度',fontsize=14,y=1.01,fontproperties=font)#使用y轴的参数调整了标题在y轴方向相对于图片顶部的位置
plt.show()#显示中文有问题，需要下载导入字体，后面做
print('--------------练习2---------------')
# 为iris数据集的每一列生成直方图
fig,ax = plt.subplots(2,2,figsize=(6,4))#在6*4英寸的地方生成2*2的4个直方图
ax[0,0].hist(df['花瓣宽度'],color='black')
ax[0,0].set_ylabel('数量', fontsize=12, fontproperties=font)
ax[0,0].set_xlabel('宽度', fontsize=12, fontproperties=font)
ax[0,0].set_title('花瓣宽度',fontsize=14,y=1.01,fontproperties=font)
ax[0,1].hist(df['花瓣长度'],color='black')
ax[0,1].set_ylabel('数量', fontsize=12, fontproperties=font)
ax[0,1].set_xlabel('宽度', fontsize=12, fontproperties=font)
ax[0,1].set_title('花瓣长度',fontsize=14,y=1.01,fontproperties=font)
ax[1,0].hist(df['花萼宽度'],color='black')
ax[1,0].set_ylabel('数量', fontsize=12, fontproperties=font)
ax[1,0].set_xlabel('宽度', fontsize=12, fontproperties=font)
ax[1,0].set_title('花萼宽度',fontsize=14,y=1.01,fontproperties=font)
ax[1,1].hist(df['花萼长度'],color='black')
ax[1,1].set_ylabel('数量', fontsize=12, fontproperties=font)
ax[1,1].set_xlabel('宽度', fontsize=12, fontproperties=font)
ax[1,1].set_title('花萼长度',fontsize=14,y=1.01,fontproperties=font)
plt.tight_layout()#自动调整子插图，避免排版拥挤
plt.show()
print('--------------练习3(散点图)---------------')
fig,ax=plt.subplots(figsize=(6,6))
ax.scatter(df['花瓣宽度'],df['花瓣长度'],color='green')
ax.set_xlabel('花瓣宽度',fontproperties=font)
ax.set_ylabel('花瓣长度',fontproperties=font)
ax.set_title('花瓣散点图',fontproperties=font)
plt.show()
print('--------------练习4(线图)---------------')
fig,ax = plt.subplots(figsize=(6,6))
ax.plot(df['花瓣长度'],color='blue')
ax.set_xlabel('样本编号',fontproperties=font)
ax.set_ylabel('花瓣长度',fontproperties=font)
ax.set_title('花瓣长度图',fontproperties=font)
plt.show()
print('--------------练习5(条形图)---------------')
fig,ax=plt.subplots(figsize=(6,6))
bar_width =.8
labels = [x for x in df.columns if '长度' in x or '宽度' in x]
ver_y = [df[df['类别']=='Iris-versicolor'][x].mean() for x in labels]
vir_y = [df[df['类别']=='Iris-virginica'][x].mean() for x in labels]
set_y = [df[df['类别']=='Iris-setosa'][x].mean() for x in labels]
x=np.arange(len(labels))#x轴个数
ax.bar(x,vir_y,bar_width,bottom=set_y,color='darkgrey')#bottom参数。这个参数将该序列的y点的最小值设置为其下面那个序列的y点的最大值（创建堆积条形图）
ax.bar(x,ver_y,bar_width,bottom=ver_y,color='white')
ax.bar(x,ver_y,bar_width,color='black')
ax.set_xticks(x+(bar_width/2))
ax.set_xticklabels(labels,rotation=-70,fontsize=12,fontproperties=font)#传入列名
ax.set_title('每个类别中特征的平均测量值',y=1.01)
ax.legend(['Virginica','Setosa','Versicolor'])#按顺序放置图例
plt.show()
