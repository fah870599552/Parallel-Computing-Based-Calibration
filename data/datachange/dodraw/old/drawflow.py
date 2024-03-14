#导入包
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
#读取文件名为hongkong的csv文件，''内是文件所在位置
df = pd.read_csv(r'D:\bishe\anzac_calibration\cab5\20190902data\20190902data\2.00-11.59flow0902.csv', sep=';')
df.head()
#只显示'local','parameter','value'列的数据
plt.rcParams['font.sans-serif'] = ['SimHei']#改字体，使标题中的中文字符可以正常显示
plt.figure(figsize=(8,6))#设置画布大小
dfx = df.iloc[:, 0]
dfy = df.iloc[:, 1]
x = dfx.values.tolist()
y = dfy.values.tolist()
plt.rcParams['font.serif'] = ['Times New Roman']
plt.style.use('seaborn-paper')
plt.title(u'9/2 2:00-11:59   820-SB field flow', fontsize=16)#设置标题
plt.xlabel(u'Time', fontsize=10)#设置x轴标签
plt.ylabel(u'flow[vehicles/hour]', fontsize=10)#设置y轴标签
plt.plot(x,y,label=u'1-Minute Flow')
ax = plt.gca() #ax为两条坐标轴的实例
x_major_locator = MultipleLocator(60)
ax.xaxis.set_major_locator(x_major_locator)
plt.ylim(0,9000)
plt.legend(loc='best')#loc也可以等于0到10，分别代表不同的位置，可以尝试
"""legend( handles=(line1, line2, line3),
           labels=('label1', 'label2', 'label3'),
           'upper right')
    The *loc* location codes are::

          'best' : 0,          (currently not supported for figure legends)
          'upper right'  : 1,
          'upper left'   : 2,
          'lower left'   : 3,
          'lower right'  : 4,
          'right'        : 5,
          'center left'  : 6,
          'center right' : 7,
          'lower center' : 8,
          'upper center' : 9,
          'center'       : 10,"""

plt.show()