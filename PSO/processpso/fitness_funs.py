#encoding: utf-8
import numpy as np
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
import json  # 数据格式
import numpy as np
import xml.etree.ElementTree
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import threading
import math
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
    #global lock
    #lock = l
def lane2edge(detector, detectorData):  # 根据同一截面不同车道的检测器数据求截面数据
    for d in detectorData:
        if detector.name in d.name:
            if np.array(detector.speed).shape[0] == 0:
                detector.speed = d.speed
                detector.occupancy = d.occupancy
                detector.flow = d.flow
            else:
                speed = detector.speed
                detector.speed = speed + d.speed
                occupancy = detector.occupancy
                detector.occupancy = occupancy + d.occupancy
                flow = detector.flow
                detector.flow = flow + d.flow
    # detector.speed = detector.speed/detector.flow #没有考虑5min没车的情况
    speed = (detector.speed / detector.flow)
    np.nan_to_num(speed)
    detector.speed = speed
    return detector


def changevalue(parameterdict):  # 改变vtype中的参数值
    # Open original file
    tree = xml.etree.ElementTree.parse('vtype.xml')
    for parameter in parameterdict.keys():
        value = parameterdict[parameter]
        for element in tree.findall('vType'):
            element.attrib[parameter] = str(value)
    tree.write('vtype.xml')
    return


# 计算C1指标
def ChangeData2(realdata, detector):
    rspeed = []
    for key in detector.keys():
        rspeed.append(realdata[key])
    speeddata = np.array(rspeed)
    return speeddata


def findsub(HBS, l, r):
    n = 0
    while l >= 0 and r <= len(HBS) and HBS[l - 1] == 1 and HBS[r + 1] == 1:
        l -= 1
        r += 1
        n += 1
        if n == 2:
            return True
            break
    if n < 2:
        return False


def BSvalue(S, Vth):  # 判断0或者为1
    i = S.shape[0]  # 数组的行，i个检测器
    t = S.shape[1]  # 数组的列， t个时间点
    BS = np.zeros((i, t))
    for j in range(i):
        for k in range(t):
            if S[j][k] < Vth:  # 拥堵点
                BS[j][k] = 1
            else:
                BS[j][k] = 0
    # 检查是否有被1包围的0
    for m in range(i):
        for n in range(t - 2):
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
    BSM = np.multiply(BSS, BSR)  # 计算BSS与BSR的交集，数组对应元素相乘
    BSP = BSS + BSR  # 计算BSS与BSR的并集，对应元素相加
    # 计算指标C1
    CU21 = 0
    CD21 = 0
    for m in range(i):
        CU11 = np.sum(BSM[m])
        CU11 = CU11 * x[m]
        CU21 += CU11
        CD11 = np.sum(BSP[m])
        CD11 = CD11 * x[m]
        CD21 += CD11
    C1 = 2 * CU21 / CD21
    # 计算指标C2
    CU22 = 0
    CD22 = 0
    SSM = abs(S - R)  # 仿真与实际的速度差的具体值
    SSP = S + R  # 仿真与实际的速度对应元素相乘
    BSM = np.multiply(SSM, BSM)
    BSP = np.multiply(SSP, BSM)
    for m in range(i):
        CU12 = np.sum(BSM[m])
        CU12 = CU12 * x[m]
        CU22 += CU12
        CD12 = np.sum(BSP[m])
        CD12 = CD12 * x[m]
        CD22 += CD12
    if CD22 == 0 and CU22 == 0:
        C2 = 0.0
    else:
        C2 = 1 - 2 * CU22 / CD22
    return C1, C2
def Objfunction(simulationdata, detector):  # 求均方根误差#这里返回计算的函数结果
    devspeed = 0
    devflow = 0
    num = 0
    with open('D:/graduate1/SUMMER/Calibration/data/datachange/Newdata/6.00-6.59speed0902.json', 'r') as load_f:
        tspeed = json.load(load_f)
    with open('D:/graduate1/SUMMER/Calibration/data/datachange/Newdata/6.00-6.59flow0902.json', 'r') as load_f:
        tflow = json.load(load_f)
    for d in simulationdata:
        d.speed = d.speed.flatten()
        for i in range(len(d.speed)):
            # print(d.speed[i])
            if math.isnan(d.speed[i]):
                d.speed[i] = d.speed[i - 1]
        devspeed += np.sum((d.speed * 3.6 - tspeed[d.name]) ** 2)
        num += len(d.speed)
        devflow += np.sum((d.flow - np.array(tflow[d.name]) / 60) ** 2)
    # print(num, devspeed, devocc)
    #    return devspeed, devocc, devflow
    dspeed = []
    for d in simulationdata:
        d.speed = d.speed.flatten()
        dspeed.append(d.speed * 3.6)
    S = np.array(dspeed)
    R = ChangeData2(tspeed, detector)
    # 计算C1C2，目标数据格式
    Vth = 45
    CC = C1C2(S, R, Vth)
    return (devspeed / num) ** 0.5, 1 - CC[0]
def subaimFunc(args):  # 仿真获取优化所需数据#参数传到这里
    # 初始数据
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
            # 进程名
    pid = mp.current_process().name
    # 修改所需优化的参数
    para = {'tau': args[0], 'lcCooperative': args[1],
            'accel': args[2], 'speedDev': args[3],
            'speedFactor': args[4], 'lcAssertive': args[5], 'decel': args[6]}  # 7个决策变量
    lock.acquire()  # lock冲突
    changevalue(para)  # 改换道跟驰
    traci.start(['sumo', "-c", "BH4calib.sumocfg", "--output-prefix", str(pid)])  # 开始仿真
    threading.Timer(3, lock.release).start()  # release in 3 second
    for step in range(0, 4200):  # 仿真循环
        traci.simulationStep()
    traci.close()  # 仿真结束
    for d in d2:  # 读取检测器数据
        d.readdata(pid, warmup, duration, interval, 1)
    for de in d1:
        de = lane2edge(de, d2)
        de.edge()  # 车道检测器→截面
    r = Objfunction(d1, detector)  # 计算目标函数值 speed，flow
    """================改=============="""
    #f = open("relength.txt", "a")  # 记录目标函数值及对应参数
    #f.writelines(str(args))
    #f.writelines(str(r) + "\n")
    #f.close()
    return r
def init(l):  # lock进程
	global lock
	lock = l

#为了便于图示观察，试验测试函数为二维输入、二维输出
#适应值函数：实际使用时请根据具体应用背景自定义
def fitness_(args):#IVars参数列表
    lock = mp.Lock()
    #目的是利用传输进来的参数集合，返回两个目标函数的值，并行待会儿再说
    pool = ProcessPool(1, initializer=init, initargs=(lock,))
    #result = subaimFunc(args)#两个目标函数的值
    result1 = pool.map_async(subaimFunc, args)  # 异步进程池非阻塞32*2的列表
    result1.wait()
    pool.close()
    result1=result1.get()
    return result1
