# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding=utf-8
# -*- coding: utf-8 -*-
""" QuickStart """
"""在脚本中引入TRACI，要使用该库，/tools 目录必须位于python加载路径上。"""
import os
import sys
 
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("please declare environment variable 'SUMO_HOME'")
     
import traci
import json#数据格式
import numpy as np
import geatpy as ea
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
        self.flow =[]

    def readdata(self, pid, warmup, duration, interval, day = 1):#从sumo输出中读取速度，流量以及占有率数据；默认是1天的数据,跑一次
        starttime = warmup
        endtime = starttime+duration
        n = int(duration/interval)
        self.speed = np.zeros([day,n])
        self.occupancy = np.zeros([day,n])
        self.flow = np.zeros([day,n])
        tree = xml.etree.ElementTree.parse(str(pid)+"out.xml")
        root = tree.getroot()
        for i in range(day):
            s = []
            o = []
            f = []
            for elem in root.iter(tag='interval'):
                if float(elem.get('begin')) >= starttime :
                    if float(elem.get('end')) <= endtime :
                        if self.name == elem.get('id'):
                            s.append(float(elem.get('speed'))*float(elem.get('nVehContrib')))
                            o.append(float(elem.get('occupancy')))
                            f.append(float(elem.get('nVehContrib')))
            self.speed[i] = s
            self.occupancy[i] = o
            self.flow[i] = f
            starttime = endtime+warmup
            endtime = starttime+duration
        '''self.speed = np.mean(self.speed, axis=0)
        self.occupancy = np.mean(self.occupancy, axis=0)
        self.flow = np.mean(self.flow, axis=0)#只跑一次仿真不需要'''
    def edge(self):
        self.occupancy = self.occupancy/self.n

def aggragate(data, starttime = 0, endtime = 1440, interval = 15 ): # 如果数据精度是1min，可以进行累计（如15min）
    result=[]
    for i in range(starttime, endtime, interval):#计算结果
        result.append(np.sum(data[i:i+interval]))
    return result

def average(data, starttime = 0, endtime = 1440, interval = 15 ): # 同理，求一段时间平均
    result=[]
    for i in range(starttime, endtime, interval):#计算结果
        result.append(np.mean(data[i:i+interval]))
    return result

def init(l):  # lock进程
	global lock
	lock = l

        
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

def changevalue(parameterdict): # 改变vtype中的参数值
    # Open original file
    tree = xml.etree.ElementTree.parse('vtype.xml')
    for parameter in parameterdict.keys():
        value = parameterdict[parameter]
        for element in tree.findall('vType'):
            element.attrib[parameter] = str(value)
    tree.write('vtype.xml')
    return
#计算C1指标
def ChangeData2(realdata, detector):
    rspeed = []
    for key in detector.keys():
        rspeed.append(realdata[key])
    speeddata = np.array(rspeed)
    return speeddata

def findsub(HBS, l, r):
    n = 0
    while l>=0 and r<=len(HBS) and HBS[l-1] == 1 and HBS[r+1] == 1:
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
        for n in range(t-2):
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
    BSM = np.multiply(SSM, BSM)
    BSP = np.multiply(SSP, BSM)
    for m in range(i):
        CU12 = np.sum(BSM[m])
        CU12 = CU12*x[m]
        CU22 += CU12
        CD12 = np.sum(BSP[m])
        CD12 = CD12 * x[m]
        CD22 += CD12
    if CD22 == 0 and CU22 == 0:
        C2 = 0.0
    else:
        C2 =1 - 2*CU22/CD22
    return C1, C2

def Objfunction(simulationdata, detector): # 求均方根误差
    devspeed = 0
    devflow = 0
    num = 0
    with open('D:/graduate1/SUMMER/Calibration/data/datachange/Newdata/6.00-6.59speed0902.json', 'r') as load_f:
        tspeed = json.load(load_f)
    with open('D:/graduate1/SUMMER/Calibration/data/datachange/Newdata/6.00-6.59flow0902.json', 'r') as load_f:
        tflow = json.load(load_f)
    for d in simulationdata:
        d.speed = d.speed.flatten()
        devspeed += np.sum((d.speed*3.6 - tspeed[d.name])**2)
        num += len(d.speed)
        devflow += np.sum((d.flow - np.array(tflow[d.name])/60)**2)
    #print(num, devspeed, devocc)
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
    return (devspeed/num)**0.5, 1-CC[0]


def subaimFunc(args): # 仿真获取优化所需数据
    # 初始数据
    warmup = 600
    duration = 3600
    runtime = 1
    interval =60
    detector = {'820 SB': 3, '819 SB': 3, '818 SB': 3, '817 SB': 3, '816 SB': 3, '815 SB': 3, '814 SB': 3, '813 SB': 3,
                '812 SB': 3, '811 SB': 3, '810 SB': 3}
    d1=[]
    d2=[]
    for key in detector.keys():
        d1.append(DetectorData(key, detector[key]))
        for j in range(detector[key]):
            d2.append(DetectorData(key+'_'+str(j+1),1))   

    # 进程名
    pid = mp.current_process().name
    # 修改所需优化的参数
    para = {'tau': args[0], 'lcCooperative': args[1],
            'accel': args[2],'speedDev': args[3],
            'speedFactor': args[4], 'lcAssertive': args[5], 'decel':args[6]}#7个决策变量

    lock.acquire() # lock冲突
    changevalue(para) # 改换道跟驰
    traci.start(['sumo', "-c", "BH4calib.sumocfg", "--output-prefix", str(pid)]) # 开始仿真
    threading.Timer(3, lock.release).start() # release in 3 second
    for step in range(0, 4200): # 仿真循环
        traci.simulationStep()
    traci.close()#仿真结束
    for d in d2: # 读取检测器数据
        d.readdata(pid, warmup, duration, interval, 1)
    for de in d1:
        de = lane2edge(de, d2)
        de.edge() # 车道检测器→截面
    r = Objfunction(d1, detector)#计算目标函数值 speed，flow
    """================改=============="""
    f=open("relength.txt","a") # 记录目标函数值及对应参数
    f.writelines(str(args))
    f.writelines(str(r)+"\n")
    f.close()
    return r

# 自定义问题类
class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self, M, PoolType):
        name = 'GA_Calibration' # 初始化name（函数名称，可以随意设置）
        maxormins = [1, 1]# 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）暂定flow，speed[1, 1]
        Dim = 7 # 初始化Dim（决策变量维数） 
        varTypes = np.array([0, 0, 0, 0, 0, 0, 0]) # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [1, 0, 2, 0, 0.5, 0.8, 1]
        ub = [4, 1, 9, 0.5, 1.5, 1.5, 7] # 决策变量上界
        lbin = [1, 0, 2, 0, 0.5, 0.8, 1]  # 决策变量下边界
        ubin = [4, 1, 9, 0.5, 1.5, 1.5, 7]#
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2) # 设置池的大小，可以包含两个线程在跑，2
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count()) # 获得计算机的核心数
            self.pool = ProcessPool(1, initializer=init, initargs=(lock,)) # 设置池的大小，，将3改为计算机的核心数
    
    def aimFunc(self, pop):
        Vars = pop.Phen # 得到决策变量矩阵#每一代里面有很多组的参数值
        lVars = list(Vars)#32*7的列表
        if self.PoolType == 'Thread':#是线程还是进程
            pop.ObjV = np.array(list(self.pool.map(subaimFunc, lVars)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subaimFunc, lVars)#异步进程池非阻塞32*2的列表
            result.wait()
            pop.ObjV = np.array(result.get())#传输到每一代的结果列表中去
        '''result = list (map(subaimFunc, lVars))
        print(result)
        pop.ObjV = np.array(result).reshape(-1,1)'''
        
    def calReferObjV(self): # 计算全局最优解
        uniformPoint, ans = ea.crtup(self.M, 10000) # 初始化种群染色体：生成10000个在各目标的单位维度上均匀分布的参考点
        globalBestObjV = uniformPoint / 2
        return globalBestObjV

if __name__ == '__main__':
    
    # 编写执行代码
    """===============================实例化问题对象=============================="""
    lock = mp.Lock()
    M = 2  
    PoolType = 'Process'                   # 设置目标维数
    problem = MyProblem(M, PoolType)    # 生成问题对象
    """==================================种群设置================================="""
    Encoding = 'RI'           # 编码方式
    NIND = 2                 # 种群规模8
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器，译码矩阵
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置==============================="""
    myAlgorithm = ea.moea_NSGA3_templet(problem, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 2  # 最大进化代数
    myAlgorithm.drawing = 2   # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制过程动画）
    """==========================调用算法模板进行种群进化=========================
        调用run执行算法模板，得到帕累托最优解集NDSet。NDSet是一个种群类Population的对象。
        NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
        详见Population.py中关于种群类的定义。
        """
    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    NDSet.save()  # 把非支配种群的信息保存到文件中
    problem.pool.close()  # 及时关闭问题类中的池，否则在采用多进程运算后内存得不到释放
    # 输出
    print('用时：%f 秒' % (myAlgorithm.passTime))
    print('评价次数：%d 次' % (myAlgorithm.evalsNum))
    print('非支配个体数：%d 个' % (NDSet.sizes))
    print('单位时间找到帕累托前沿点个数：%d 个' % (int(NDSet.sizes // myAlgorithm.passTime)))
    # 计算指标
    if myAlgorithm.log is not None and NDSet.sizes != 0:
        print('GD', myAlgorithm.log['gd'][-1])
        print('IGD', myAlgorithm.log['igd'][-1])
        print('HV', myAlgorithm.log['hv'][-1])
        print('Spacing', myAlgorithm.log['spacing'][-1])
        """=========================进化过程指标追踪分析========================="""
        metricName = [['igd'], ['hv']]
        Metrics = np.array([myAlgorithm.log[metricName[i][0]] for i in range(len(metricName))]).T
        # 绘制指标追踪分析图
        ea.trcplot(Metrics, labels=metricName, titles=metricName)






