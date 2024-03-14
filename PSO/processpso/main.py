# encoding: utf-8
import numpy as np
from mopso import *
import datetime
#import cProfile, pstats, io
import multiprocessing as mp

def main():
    starttime = datetime.datetime.now()
    w = 1  # 惯性因子
    c1 = 2  # 局部速度因子
    c2 = 2  # 全局速度因子
    particals = 2  # 粒子群的数量
    cycle_ = 2  # 迭代次数
    mesh_div = 10  # 网格等分数量
    thresh = 300  # 外部存档阀值
    min_ = np.array([1, 0, 2, 0, 0.5, 0.8, 1])  # 粒子坐标的最小值
    max_ = np.array([4, 1, 9, 0.5, 1.5, 1.5, 7])  # 粒子坐标的最大值
    #pr = cProfile.Profile()
    #pr.enable()
    mopso_ = Mopso(particals, w, c1, c2, max_, min_, thresh, mesh_div)  # 粒子群实例化
    pareto_in, pareto_fitness = mopso_.done(cycle_)  # 经过cycle_轮迭代后，pareto边界粒子
    endtime = datetime.datetime.now()
    print('运行时间')
    print((endtime - starttime).seconds)
    #pr.disable()
    #s = io.StringIO()
    #sortby = "cumtime"  # 仅适用于 3.6, 3.7 把这里改成常量了
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())
    #pr.dump_stats("pipeline5.prof")
    np.savetxt("./img_txt/pareto_in.txt", pareto_in)  # 保存pareto边界粒子的坐标
    np.savetxt("./img_txt/pareto_fitness.txt", pareto_fitness)  # 打印pareto边界粒子的适应值
    print
    "\n", "pareto边界的坐标保存于：/img_txt/pareto_in.txt"
    print
    "pareto边界的适应值保存于：/img_txt/pareto_fitness.txt"
    print
    "\n", "迭代结束,over"
if __name__ == "__main__":
    main()
