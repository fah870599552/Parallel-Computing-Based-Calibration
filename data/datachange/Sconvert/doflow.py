import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math
#时间序列生成函数
def get_date_list(begin_date,end_date):
    date_list = [x.strftime('%H:%M') for x in list(pd.date_range(start=begin_date, end=end_date, freq='min'))]
    return date_list
#绘图函数
#绘制输入柱状图
def drawpic(data, time1, time2, str):
    font1 = {'family': 'Times New Roman', 'size': 14}
    #指定字体为SimHei，用于显示中文，如果Ariel,中文会乱码
    mpl.rcParams["axes.unicode_minus"]=False
    #用来正常显示负号
    times = get_date_list(time1, time2)
    ll = int(len(times)/5)
    x = [ i for i in range(ll)]
    y = data
    #数据
    labels=times
    #定义柱子的标签
    plt.bar(x, y, align="center", color="blue", ec='gray')
    #绘制纵向柱状图,hatch定义柱图的斜纹填充，省略该参数表示默认不填充。
    #bar柱图函数还有以下参数：
    #颜色：color,可以取具体颜色如red(简写为r),也可以用rgb让每条柱子采用不同颜色。
    #描边：edgecolor（ec）：边缘颜色；linestyle（ls）：边缘样式；linewidth（lw）：边缘粗细
    #填充：hatch，取值：/,|,-,+,x,o,O,.,*
    #位置标志：tick_label
    plt.xlabel("Time", fontdict=font1)
    plt.ylabel("vechperhour", fontdict=font1)
    title = time1+'-'+time2+str
    plt.title(title, fontdict=font1)
    filename = './flow/1/'+ str + '.jpg'
    plt.savefig(filename, dpi=500)
    return None
#绘制双折线图
def drawpic2(data1, data2, time1, time2, str):
    font1 = {'family': 'Times New Roman', 'size': 14}
    mpl.rcParams["axes.unicode_minus"] = False
    # 用来正常显示负号
    times = get_date_list(time1, time2)
    ll = int(len(times) / 5)
    x = [i for i in range(ll)]
    y1 = data1
    y2 = data2
    plt.title(u'0902 5:00-7:59  810-SB flow', fontdict=font1)  # 设置标题
    plt.plot(x, y1, color="red", label=u'Field Flow', linewidth='3', marker='o', markersize='6')
    plt.plot(x, y2, color="blue", label=u'Input Flow', linestyle='-.', linewidth='3', marker='*', markersize='8')
    plt.legend()
    plt.xlabel(u'Time[5 min]', fontdict=font1)  # 设置x轴标签
    plt.ylabel(u'flow[vehicles/hour]', fontdict=font1)  # 设置y轴标签
    ax = plt.gca()  # ax为两条坐标轴的实例
    x_major_locator = MultipleLocator(5)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(2000, 9000)
    plt.legend(loc='best')  # loc也可以等于0到10，分别代表不同的位置，可以尝试
    filename = './flow/1/doubleline.jpg'
    plt.savefig(filename, dpi=500)
    return None
#聚合函数
def aggragate(data, starttime = 0, endtime = 1440, interval = 5, ll=250):
    result=[]
    for i in range(0, ll, interval):
            print(data[i])#计算结果
            print(data[i:i+interval])
            result.append(np.sum(data[i:i+interval]))
    return result
#打开相关文件
data = pd.DataFrame(pd.read_csv('D://graduate1//SUMMER//Calibration//data//datachange//Newdata//5.50-6.59flow0902.csv', sep=";"))
ll = 70
print(data)
#检测器位置函数
p = [554, 1300]
#delay
delay = [0, 0]#路段最大限速为27.78km/h
#up与down构造检测器构造
R1D = '820 SB'
Off_up = ['817 SB', '813 SB']
Off_down = ['816 SB', '810 SB']
#flow数据结构构造与计算
R1_total = data.iloc[:, 1]
R1_total = np.array(R1_total.values.tolist())
R1_total = aggragate(R1_total, 0, ll, 5, ll)
total = data.iloc[:, 11]
total = np.array(total.values.tolist())
total = aggragate(total, 0, ll, 5, ll)
ofin_flow = []
data = data.to_dict()
for i in range(len(Off_up)):
    data_up =  np.array(list(data[Off_up[i]].values()))
    data_down = np.array(list(data[Off_down[i]].values()))
    a = aggragate(data_up, 0, ll, 5, ll)#上游检测器所有流量数据，start为290（？）+delay(eg.10min),end为600+delay(eg.10min),5min聚合，这个volum为1min经过的车辆
    b = aggragate(data_down, 0+delay[i], ll+delay[i], 5, ll)#下游检测器所有流量数据，start为290（？）+delay(eg.10min),end为600+delay(eg.10min),5min聚合，这个volum为1min经过的车辆
    if i == 0:
        l = [a[j]- b[j]for j in range(len(a))]
    else:
        l = [b[j] - a[j] for j in range(len(a))]
    ofin_flow.append(np.array(l))
mm = int(ll/5)
R1 = [R1_total[i]-ofin_flow[0][i] for i in range(mm)]
P = len(R1)
for i in range(P):
    R1[i] = R1[i]/10
    ofin_flow[0][i] = ofin_flow[0][i]/5
    ofin_flow[1][i] = ofin_flow[1][i]/5
    R1_total[i] = R1_total[i]/5
    total[i] = total[i]/5
#流量调节

#SUMO数据写入打印
date = '0902'
filename = '1mycalibrator'+date+'.xml'
with open(filename, 'a') as file_object:
    file_object.write('<additional>\n')

with open(filename, 'a') as file_object:
    file_object.write('\t<calibrator id="A1" lane="444071875_0" pos="1.00" out="result.xml">\n')
for i in range(mm):
    with open(filename, 'a') as file_object:
        file_object.write('\t\t<flow begin="%d" end=\"%d\" id="444071875_M2.4%d" vehsPerHour=\"%d\" route="444071875_M2.4" type="carr" />\n'%(i*300, (i+1)*300, i, max(0, R1[i])))
with open(filename, 'a') as file_object:
    file_object.write('\t</calibrator>\n')
with open(filename, 'a') as file_object:
    file_object.write('\t<calibrator id="A2" lane="444071875_1" pos="1.00" out="result.xml">\n')
for i in range(mm):
    with open(filename, 'a') as file_object:
        file_object.write('\t\t<flow begin="%d" end=\"%d\" id="444071875_M2.4%d" vehsPerHour=\"%d\" route="444071875_M2.4" type="carr" />\n'%(i*300, (i+1)*300, i, max(0, R1[i])))#此时间隔为5分钟
with open(filename, 'a') as file_object:
    file_object.write('\t</calibrator>\n')
with open(filename, 'a') as file_object:
    file_object.write('\t<calibrator id="B1" lane="444071875_2" pos="1.00" out="result.xml">\n')
for i in range(mm):
    with open(filename, 'a') as file_object:
        file_object.write('\t\t<flow begin="%d" end=\"%d\" id="444071875_Off3%d" vehsPerHour=\"%d\" route="444071875_Off3" type="carr" />\n'%(i*300, (i+1)*300, i, max(0, ofin_flow[0][i])))#此时间隔为5分钟
with open(filename, 'a') as file_object:
    file_object.write('\t</calibrator>\n')
with open(filename, 'a') as file_object:
    file_object.write('\t<calibrator id="C1" edge="On3" pos="1.00" out="result.xml">\n')
for i in range(mm):
    with open(filename, 'a') as file_object:
        file_object.write('\t\t<flow begin="%d" end=\"%d\" id="On3_M2.4%d" vehsPerHour=\"%d\" route="On3_M2.4" type="carr" />\n'%(i*300, (i+1)*300, i, max(0, ofin_flow[1][i])))#此时间隔为5分钟

with open(filename, 'a') as file_object:
    file_object.write('\t</calibrator>\n')
with open(filename, 'a') as file_object:
    file_object.write('</additional>\n')
#R1流量输入柱状图
time1 = '5:50'
time2 = '6:59'
R11 = drawpic(R1, time1, time2, str = 'R1')
R22 = drawpic(ofin_flow[0], time1, time2, str = 'R2')
R33 = drawpic(ofin_flow[1], time1, time2, str = 'R3')
#R44 = drawpic(R1+ofin_flow[0], time1, time2, str = '计算的820')
#R66 = drawpic(R1_total, time1, time2, str = 'field820')
#R55 = drawpic(R1+ofin_flow[1], time1, time2, str = '计算的810')
#R77 = drawpic(total, time1, time2, str = 'field810')
R88 = drawpic2(total, R1+ofin_flow[1], time1, time2, str = '810-SB')
#R2流量输入柱状图
#R3流量输入柱状图