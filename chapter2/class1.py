import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

print('--------------练习1---------------')
pd.set_option('display.max_columns', 30)  # 显示最大行数和列数，如果超额就显示省略号
pd.set_option('display.max_colwidth', 100)  # 列长度
pd.set_option('display.precision', 3)  # 显示小数点后几位

df = pd.read_csv('magic.csv')
df.columns  # 输出列标题
print(df.columns)
print(df.head(3).T)  # .T语法将转置我们的数据框并垂直显示它
print('--------------练习2---------------')
mu = df[df['listingtype_value'].str.contains('Apartments For')]# 多房间
su = df[df['listingtype_value'].str.contains('Apartment For')]# 单一单元
# print(len(mu))
# print(len(su))
print('--------------练习3---------------')
# print(su['propertyinfo_value'])# 包含卧室浴室的数量以及平方英尺的列
# 检查没有包含'bd'，或'studio'（卧室）的行数
leng=len(su[~(su['propertyinfo_value'].str.contains('Studio')|su['propertyinfo_value'].str.contains('bd'))])
print(leng)#输出0   '～'是取反符
# 查看没有包含'ba'（浴室）的行数
len_ba = len(su[~(su['propertyinfo_value'].str.contains('ba'))])
print(len_ba)# 输出6
# 有少部分房屋缺少浴室数量信息
print('--------------练习4(填充或插补缺失的数据点)---------------')
no_baths = su[~(su['propertyinfo_value'].str.contains('ba'))]# 选择缺少浴室信息的房源
sucln = su[~su.index.isin(no_baths.index)]# isin检查是否存在参数索引中
# print(sucln)

def parse_info(row):
    if not 'sqft' in row:
        br,ba=row.split('•')[:2]
        sqft = np.nan
    else:
        br,ba,sqft=row.split('•')[:3]
    return pd.Series({'卧室':br,'浴室':ba,'平方数':sqft})

attr = sucln['propertyinfo_value'].apply(parse_info)
#在取值中将字符串（'ba','bd'）删除
attr_cln = attr.applymap(lambda x:x.strip().split(' ')[0] if isinstance(x,str) else np.nan)

sujnd = sucln.join(attr_cln)
# print(sujnd.T)
print('--------------练习5(处理建筑物楼层)---------------')
def parse_addy(r):
    so_zip = re.search('NY(\d+)',r)
    so_flr = re.search('(?:APT|#)\s+(\d+)[A-Z]+,',r)
    if so_zip:
        zipc = so_zip.group(1)
    else:
        zipc = np.nan

    if so_flr:
        flr = so_flr.group(1)
    else:
        flr = np.nan

    return pd.Series({'邮编':zipc,'楼层':flr})

flrzip = sujnd['routable_link/_text'].apply(parse_addy)

suf = sujnd.join(flrzip)
print(suf.T)
print('--------------练习6(将数据减少为感兴趣的那些列)---------------')
sudf = suf[['pricelarge_value_prices','卧室','浴室','平方数','楼层','邮编']]
#
sudf.rename(columns={'pricelarge_value_prices':'租金'},inplace=True)
sudf.reset_index(drop=True,inplace=True)
print(sudf)
print('--------------练习7(分析数据)---------------')
print(sudf.describe())
# .loc 通过行标签索引行数据
sudf.loc[:,'卧室'] = sudf['卧室'].map(lambda x: 0 if 'Studio' in x else x)
print(sudf)
print('--------------练习8(解决列中数据类型问题)---------------')
sudf.loc[:,'租金'] = sudf['租金'].astype(int)
sudf.loc[:,'卧室'] = sudf['卧室'].astype(int)
sudf.loc[:,'浴室'] = sudf['浴室'].astype(float)#存在半间浴室的情况，所以使用浮点型
sudf.loc[:,'平方数'] = sudf['平方数'].str.replace(',','')#存在Nans，将逗号去掉
sudf.loc[:,'平方数'] = sudf['平方数'].astype(float)
sudf.loc[:,'楼层'] = sudf['楼层'].astype(float)
sudf=sudf.drop([318])#索引318是有问题的房源，删除
print(sudf.info())
print(sudf.describe())
print('--------------练习9(pivot_table透视图)---------------')
sudf1=sudf.pivot_table('租金','邮编','卧室',aggfunc='mean')
print(sudf1)
sudf2=sudf.pivot_table('租金','邮编','卧室',aggfunc='count')
print(sudf2)
print('--------------练习10(可视化数据----没有找到ync.json，稍后找)---------------')
su_lt_two = sudf[sudf['卧室']<2]# 将房源聚焦到一居室和工作室中
import folium

map = folium.Map(location=[40.748817,-73.985428],zoom_start=13)#创建Map对象，传入坐标和缩放级别
# map.geo_json(geo_path='')
print('--------------练习11(对数据建模)---------------')
import statsmodels.api as sm
import patsy

print('参数信息（从左到右）：变量，变量在模型中的系数，标准误差，t统计量，t统计量的p值，95%的置信区间')
print('看p值这一列，可以确定独立变量从统计的角度来看是否具有意义')
# 邮编和卧室数量如何影响出租价格
f = '租金 ~ 邮编 + 卧室'# 波浪线左边，也就是租金，叫反应或因变量，右边是独立或预测变量，就是邮编和卧室
# 返回一个数据框，X矩阵由预测变量组成，y向量由相应变量组成
y,X = patsy.dmatrices(f,su_lt_two,return_type='dataframe')# 公式和包含相应列名的数据库一起传递给patsy.dmatrices()
results = sm.OLS(y,X).fit() #调用.fit（）来运行我们的模型
print(results.summary())
# 显著性，仅仅使用卧室数量和邮政编码就已经能够解释三分之一的价格差异
print('--------------练习12(预测(predict报错未解决))---------------')
print(X.head())
to_pred_idx = X.iloc[0].index# 使用X矩阵索引
print(to_pred_idx)
to_pred_zeros = np.zeros(len(to_pred_idx))#零填充数据
tpdf=pd.DataFrame(to_pred_zeros,index=to_pred_idx,columns=['value'])
print(tpdf)
# 填入一些实际的值
tpdf.loc['intercept'] = 1# 线性回归，截距必须设置为1才会返回正确的统计值
tpdf.loc['卧室'] = 1
tpdf.loc['邮编[T.10009]'] = 1
print(tpdf)
# results.predict[tpdf['value']]
tpdf['value'] = 0
tpdf.loc['Intercept'] = 1
tpdf.loc['卧室'] = 2
tpdf.loc['邮编[T.10009]'] = 1
print(tpdf)
print("---------")
results.predict(tpdf['value'])
print(tpdf)





