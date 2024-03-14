import json
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
#首先实现一个时间点的数据可视化，120个时间点
with open('D://graduate1//SUMMER//Calibration//data//datachange//Newdata//6.00-6.59speed0902.json', 'r') as load_f:
    tspeed = json.load(load_f)
    #print(tspeed.keys())

#with open('D:/bishe/anzac_calibration/fielddata/Flow0902.json', 'r') as load_f:
#   tflow = json.load(load_f)
#position_list = ['0.25', '0.5', '0.75', '1', '1.25', '1.5', '1.75', '2', '2.25', '2.5', '2.75', '3', '3.25', '3.5', '3.75', '4', '4.25', '4.5', '4.75', '5']#24/20
#lane_list = tspeed['Time']#7/5
# 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
#X 为1*200的数组
nn = 60
def get_date_list(begin_date,end_date):
    date_list = [x.strftime('%H:%M') for x in list(pd.date_range(start=begin_date, end=end_date, freq='min'))]
    return date_list
times = get_date_list('6:00', '6:59')
list_x = [[i for i in range(nn)] for j in range(11)]#12行120列
X = np.array(list_x)
detector_y = [0, 500, 450, 505,
              392, 465, 432, 555, 510, 660, 372]  # 检测器间隔
#0, 480,
detector_Y = []
yy = 0

for i in range(11):
    yy += detector_y[i]
    detector_Y .append([yy]*nn)
Y = np.array(detector_Y)
Z = np.zeros((11, nn))#12行120列
for i in range(nn):
    Z[10][i] = tspeed['820 SB'][i]
    Z[9][i] = tspeed['819 SB'][i]
    Z[8][i] = tspeed['818 SB'][i]
    Z[7][i] = tspeed['817 SB'][i]
    Z[6][i] = tspeed['816 SB'][i]
    Z[5][i] = tspeed['815 SB'][i]
    Z[4][i] = tspeed['814 SB'][i]
    Z[3][i] = tspeed['813 SB'][i]
    Z[2][i] = tspeed['812 SB'][i]
    Z[1][i] = tspeed['811 SB'][i]
    Z[0][i] = tspeed['810 SB'][i]
#Y 为1*4的数组
#Z是（200， 4）的数组
# 填充等高线
print(Z[0])
font1 = {'family': 'Times New Roman', 'size': 30}
plt.rcParams['font.size'] = 20
#levels = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]

cf = plt.contourf(X, Y, Z, 500, vmin=15, vmax= 115, cmap="jet_r")
cb = plt.colorbar(cf)
tick_locator = ticker.MaxNLocator(nbins=10)
cb.locator = tick_locator
cb.update_ticks()
plt.xlabel("Time [min]", fontdict=font1)
plt.ylabel("Location [m]", fontdict=font1)
plt.title('Speed [km/h] contour plot', fontdict=font1)
# 显示图表
plt.show()

"""bar3D = (
    Bar3D()
    .add(
        "",
        data,
        xaxis3d_opts=opts.Axis3DOpts(position_list, type_="category"),
        yaxis3d_opts=opts.Axis3DOpts(lane_list, type_="category"),
        zaxis3d_opts=opts.Axis3DOpts(type_="value"),
    )
)

bar3D.render("fieldspeed0902.html")"""