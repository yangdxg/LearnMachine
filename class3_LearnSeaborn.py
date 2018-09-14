# 专门为统计可视化而创建的库，和pandas数据框完美的协作
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 解决显示中文问题
from matplotlib.font_manager import FontManager, FontProperties

font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
df = pd.read_csv('iris.data', names=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类别'])
sns.pairplot(df, hue='类别')
plt.show()
print('--------------练习2(小提琴图)---------------')

fig, ax = plt.subplots(2, 2, figsize=(7, 7))
sns.set(style='white', palette='muted')
sns.violinplot(x=df['类别'], y=df['花萼长度'], ax=ax[0, 0])
sns.violinplot(x=df['类别'], y=df['花萼宽度'], ax=ax[0, 1])
sns.violinplot(x=df['类别'], y=df['花瓣长度'], ax=ax[1, 0])
sns.violinplot(x=df['类别'], y=df['花瓣宽度'], ax=ax[1, 1])
fig.suptitle('Violin Plots', fontsize=16, y=1.03)
for i in ax.flat:
    plt.setp(i.get_xticklabels(), rotation=-90)
fig.tight_layout()
plt.show()


