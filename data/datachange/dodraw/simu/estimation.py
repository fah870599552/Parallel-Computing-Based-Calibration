import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
sys.path.append(os.path.join('c:', os.sep, 'whatever', 'path', 'to', 'sumo', 'tools'))
from sumolib import checkBinary  # noqa
import optparse
import json  # 数据格式
import random  #
import numpy as np
import xml.etree.ElementTree
import multiprocessing as mp
import math
import matplotlib.pyplot as plt
from matplotlib import ticker

class DetectorData(object):

    def __init__(self, name, NumberOfLanes):
        self.name = name
        self.n = NumberOfLanes
        self.speed = []
        self.occupancy = []
        self.flow = []

    def readdata(self, pid, warmup, duration, interval, day=1):  # 从sumo输出中读取速度，流量以及占有率数据；默认是1天的数据,跑一次
        starttime = warmup
        endtime = starttime + duration
        n = int(duration / interval)
        self.speed = np.zeros([day, n])
        self.occupancy = np.zeros([day, n])
        self.flow = np.zeros([day, n])
        tree = xml.etree.ElementTree.parse(str(pid) + "out.xml")
        root = tree.getroot()
        for i in range(day):
            s = []
            o = []
            f = []
            for elem in root.iter(tag='interval'):
                if float(elem.get('begin')) >= starttime:
                    if float(elem.get('end')) <= endtime:
                        if self.name == elem.get('id'):
                            s.append(float(elem.get('speed')) * float(elem.get('nVehContrib')))
                            o.append(float(elem.get('occupancy')))
                            f.append(float(elem.get('nVehContrib')))
            self.speed[i] = s
            self.occupancy[i] = o
            self.flow[i] = f
            starttime = endtime + warmup
            endtime = starttime + duration
        '''self.speed = np.mean(self.speed, axis=0)
        self.occupancy = np.mean(self.occupancy, axis=0)
        self.flow = np.mean(self.flow, axis=0)#只跑一次仿真不需要'''

    def edge(self):
        self.occupancy = self.occupancy / self.n


def aggragate(data, starttime=0, endtime=1440, interval=15):  # 如果数据精度是1min，可以进行累计（如15min）
    result = []
    for i in range(starttime, endtime, interval):  # 计算结果
        result.append(np.sum(data[i:i + interval]))
    return result


def average(data, starttime=0, endtime=1440, interval=15):  # 同理，求一段时间平均
    result = []
    for i in range(starttime, endtime, interval):  # 计算结果
        result.append(np.mean(data[i:i + interval]))
    return result

#def init(l):  # lock进程
#	global lock
#	lock = l

def lane2edge(detector, detectorData): # 根据同一截面不同车道的检测器数据求截面数据
    for d in detectorData:
        if detector.name in d.name:
            if np.array(detector.speed).shape[0] == 0:
                detector.speed = d.speed
                detector.occupancy = d.occupancy
                detector.flow = d.flow
            else:
                speed = detector.speed
                detector.speed = speed+d.speed
                occupancy = detector.occupancy
                detector.occupancy = occupancy+d.occupancy
                flow = detector.flow
                detector.flow = flow+d.flow
    #detector.speed = detector.speed/detector.flow #没有考虑5min没车的情况
    speed = (detector.speed/detector.flow)
    np.nan_to_num(speed)
    detector.speed = speed
    return detector

def Objfunction(simulationdata): # 求均方根误差
    devspeed = 0
    devflow = 0
    num = 0
    for d in simulationdata:
        d.speed = d.speed.flatten()
        devspeed += np.sum((d.speed*3.6 - tspeed[d.name])**2)
        num += len(d.speed)
        devflow += np.sum((d.flow - np.array(tflow[d.name])/60)**2)
    #print(num, devspeed, devocc)
#    return devspeed, devocc, devflow
    return (devspeed/num)**0.5, (devflow/num)**0.5

def ChangeData1(simulationdata):
    #C1C2
    dspeed = []
    for d in simulationdata:
        d.speed = d.speed.flatten()
        dspeed.append(d.speed*3.6)
    speeddata = np.array(dspeed)
    #速度时空图
    return speeddata
def ChangeData2(realdata, detector):
    rspeed = []
    for key in detector.keys():
        rspeed.append(realdata[key])
    speeddata = np.array(rspeed)
    return speeddata

def findsub(HBS, l, r):
    n = 0
    while l-2>=0 and r<=len(HBS)-2 and HBS[l-1] == 1 and HBS[r+1] == 1:
        l-=1
        r+=1
        n+=1
        if n == 2:
            return True
            break
    if n<2:
        return False

def BSvalue(S, Vth):#判断0或者为1
    i = S.shape[0]  # 数组的行，i个检测器
    t = S.shape[1]  # 数组的列， t个时间点
    BS = np.zeros((i, t))
    for j in range(i):
        for k in range(t):
            if S[j][k] < Vth:#拥堵点
                BS[j][k] = 1
            else:
                BS[j][k] = 0
    #检查是否有被1包围的0
    for m in range(i):
        for n in range(t-1):
            if BS[m][n] == 0:
                if findsub(BS[m], n, n):
                    BS[m][n] = 1
    return BS
# 判断完所有检测器以及时间点的BS值
# 滑动窗口算法排查是否有被0包围的1

def C1C2(S, R, Vth):  # 输入所有检测器，所有时间点获取的速度平均值，一个数组（i, t）
# 生成BSS,BSR
    BSS = BSvalue(S, Vth)
    BSR = BSvalue(R, Vth)
    i = S.shape[0]  # 数组的行，i个检测器
    # 计算生成X间隔矩阵
    x = [372, 660, 510, 555, 432, 465, 392, 505, 450, 500, 400]
    # 计算评价指标C1
    BSM = np.multiply(BSS, BSR)#计算BSS与BSR的交集，数组对应元素相乘
    BSP = BSS + BSR#计算BSS与BSR的并集，对应元素相加
    #计算指标C1
    CU21 = 0
    CD21 = 0
    for m in range(i):
        CU11 = np.sum(BSM[m])
        CU11 = CU11*x[m]
        CU21 += CU11
        CD11 = np.sum(BSP[m])
        CD11 = CD11*x[m]
        CD21 += CD11
    C1 = 2*CU21/CD21
    #计算指标C2
    CU22 = 0
    CD22 = 0
    SSM = abs(S-R)#仿真与实际的速度差的具体值
    SSP = S+R#仿真与实际的速度对应元素相乘
    BBSM = np.multiply(SSM, BSM)
    BBSP = np.multiply(SSP, BSM)
    for m in range(i):
        CU12 = np.sum(BBSM[m])
        CU12 = CU12*x[m]
        CU22 += CU12
        CD12 = np.sum(BBSP[m])
        CD12 = CD12 * x[m]
        CD22 += CD12
    if CD22 == 0 and CU22 == 0:
        C2 = 0
    else:
        C2 =1 - 2*CU22/CD22
    return C1, C2

if __name__ == '__main__':
     #获取最优的参数集结果
     # 初始数据
     #lock = mp.Lock()
     warmup = 600
     duration = 3600
     runtime = 1
     interval = 60
     detector = {'820 SB': 3, '819 SB': 3, '818 SB': 3, '817 SB': 3, '816 SB': 3, '815 SB': 3, '814 SB': 3, '813 SB': 3,
                 '812 SB': 3, '811 SB': 3, '810 SB': 3}
     d1 = []
     d2 = []
     for key in detector.keys():
         d1.append(DetectorData(key, detector[key]))
         for j in range(detector[key]):
             d2.append(DetectorData(key + '_' + str(j + 1), 1))
     pid = mp.current_process().name
     #para = {'sigma': 1, 'tau': 1,
     #        'accel': 1, 'speedDev': 1,
      #       'speedFactor': 1, 'minGap': 1, 'length': 1}  # 7个决策变量#直接替换为最佳参数的值
     #lock.acquire()  # lock冲突
     #changevalue(para)  # 改换道跟驰
     traci.start(['sumo', "-c", "BH4calib.sumocfg", "--output-prefix", str(pid)])  # 开始仿真
     #threading.Timer(3, lock.release).start()  # release in 3 second
     for step in range(0, 4200):  # 仿真循环
         traci.simulationStep()
     traci.close()  # 仿真结束
     for d in d2:  # 读取检测器数据
         d.readdata(pid, warmup, duration, interval, 1)
     for de in d1:
         de = lane2edge(de, d2)
         de.edge()  # 车道检测器→截面
     #读取实地数据
     with open('D:/graduate1/SUMMER/Calibration/data/datachange/Newdata/6.00-6.59speed0902.json', 'r') as load_f:
         tspeed = json.load(load_f)
     with open('D:/graduate1/SUMMER/Calibration/data/datachange/Newdata/6.00-6.59flow0902.json', 'r') as load_f:
         tflow = json.load(load_f)
     # 计算目标函数值 speed，flow#这里获得了怎样的数据呢
     r = Objfunction(d1)
     print(r)
     #记录下所有的速度，流量检测值
     S = ChangeData1(d1)
     R = ChangeData2(tspeed, detector)
     #计算C1C2，目标数据格式
     Vth = 35
     CC = C1C2(S, R, Vth)
     print(CC)
     #绘制速度时空图
     # X 为1*200的数组
     nn = 60
     list_x = [[i for i in range(nn)] for j in range(11)]  # 11行120列
     X = np.array(list_x)
     detector_y = [0, 500, 450, 505,
                   392, 465, 432, 555, 510, 660, 372]  # 检测器间隔
     #  x = [372, 660, 510, 555, 432, 465, 392, 505, 450, 500, 400]
     detector_Y = []
     yy = 0
     for i in range(11):
         yy += detector_y[i]
         detector_Y.append([yy] * nn)
     Y = np.array(detector_Y)
     Z = np.zeros((11, nn))  # 12行120列
     for i in range(nn):
         for j in range(11):
             Z[10-j][i] = S[j][i]

     print(Z[0])
     # Y 为1*4的数组
     # Z是（200， 4）的数组
     # 填充等高线
     font1 = {'family': 'Times New Roman', 'size': 30}
     plt.rcParams['font.size'] = 14
     # levels = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
     cf = plt.contourf(X, Y, Z, 500, vmin=15, vmax=115, cmap="jet_r")
     cb = plt.colorbar(cf)
     tick_locator = ticker.MaxNLocator(nbins=10)
     cb.locator = tick_locator
     cb.update_ticks()
     plt.xlabel("Time [min]", fontdict=font1)
     plt.ylabel("Location [m]", fontdict=font1)
     plt.title('Speed [km/h] contour plot',fontdict=font1)
     # 显示图表
     plt.show()

