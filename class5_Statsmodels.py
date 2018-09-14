# 建模和评估学习------第一个库Statsmodels
# Statsmodels用于探索数据，估计模型，并运行统计检验的Python包
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('iris.data',names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','类别'])
print('--------------练习1(Statsmodels构建一个简单的线性回归模型)---------------')
# 对花萼长度和花萼宽度之间的关系进行建模
# 1. 先通过散点图来目测这俩者的关系
fig,ax = plt.subplots(figsize=(7,7))
ax.scatter(df['花萼宽度'][:50],df['花萼长度'][:50])
ax.set_xlabel('花萼宽度')
ax.set_ylabel('花萼长度')
ax.set_title('花萼宽度，长度，关系模型',fontsize=14,y=1.02)
plt.show()
# 线性回归，该模型的各市为Y = B0 + B1X, 其中B0为截距，B1是回归系数
# 输出后，最终公式为 花萼长度=2.6447 + 0.6909*花萼宽度
# 2.在这个数据集上运行一个线性回归模型，预估这种关系强度
y = df['花萼长度'][:50]
x = df['花萼宽度'][:50]
X = sm.add_constant(x)
results = sm.OLS(y,X).fit()
print(results.summary())
# 3. 使用结果对象来绘制回归线
fig,ax = plt.subplots(figsize=(7,7))
ax.plot(x,results.fittedvalues,label='regression line')
ax.scatter(x,y,label = 'data point',color='r')
ax.set_ylabel('花萼长度')
ax.set_xlabel('花萼宽度')
ax.set_title('花萼宽度，长度，关系模型',fontsize=14,y=1.02)
ax.legend(loc=2)
plt.show()