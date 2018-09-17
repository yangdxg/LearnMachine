# Python机器学习中的王者，为十几个算法提供了同意的API接口，建立在Python科学栈的核心模块之上
# 覆盖了分类，回归，聚类，降维，模型选择和预处理
# 打造机器学习模型第一步，理解数据应该如何构建
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier#随机森林分类起
from sklearn.model_selection import train_test_split#将数据分成训练组和测试组，会打乱数据的先后顺序

# 从csv文件导入数据,并添加标题行
df = pd.read_csv('iris.data',names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','类别'])

print('--------------练习1---------------')
clf = RandomForestClassifier(max_depth=5,n_estimators=10)#使用10个决策树的森林，每棵树最多允许五层的判定深度（避免过于拟合）
x = df.ix[:,:4]#X矩阵(df中取前四列，即花萼长度，花萼宽度，花瓣长度，花瓣宽度)
y = df.ix[:,4]#y向量(df中取第四列，即花的类别名称)
#该方法将数据打乱并划分为四个子集
# test_size被设置为0.3（数据集中的百分之30分配给测试组X_test和y_test，其余的分配给训练组X_train和y_train）
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=.3)
clf.fit(X_train,y_train)#将向量传递到指定分类器的.fit方法，
y_pred = clf.predict(X_test)#用测试数据调用分类器的预测方法
rf = pd.DataFrame(list(zip(y_pred,y_test)),columns=['predicted','actual'])#创建实际标签于预估标签的数据库框
rf['correct'] = rf.apply(lambda r:1 if r['predicted']==r['actual'] else 0,axis=1)
print(rf)
#打印准确度
print(rf['correct'].sum()/rf['correct'].count())
print('--------------练习2---------------')
# .feature_importances_ 返回特征在决策树中划分叶子节点的相对能力
# 如果一个特征能够将分组一致性的干净拆分成不同的类别，那么它将具有很高的特征重要性，这个数字的总和将始终是1
f_importance = clf.feature_importances_
f_names = df.columns[:4]
f_std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
zz = zip(f_importance,f_names,f_std)
zzs = sorted(zz,key=lambda x:x[0],reverse=True)
imps = [x[0] for x in zzs]
labels = [x[1] for x in zzs]
errs = [x[2] for x in zzs]
plt.bar(range(len(f_importance)),imps,color='r',yerr= errs,align='center')
plt.xticks(range(len(f_importance)),labels)
plt.show()
print('--------------练习3(切换分类起并使用支持向量机)---------------')
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

clf = OneVsRestClassifier(SVC(kernel='linear'))#引入支持向量机SVM而不是随机森林
x = df.ix[:,:4]
y = np.array(df.ix[:,4]).astype(str)#标签y和随机森林有区别

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=.3)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)#用测试数据调用分类器的预测方法
rf = pd.DataFrame(list(zip(y_pred,y_test)),columns=['predicted','actual'])#创建实际标签于预估标签的数据库框
rf['correct'] = rf.apply(lambda r:1 if r['predicted']==r['actual'] else 0,axis=1)
print(rf)
print(rf['correct'].sum()/rf['correct'].count())