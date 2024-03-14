# encoding: utf-8
import numpy as np
import random
class mesh_crowd(object):
    def __init__(self, curr_archiving_in, curr_archiving_fit, mesh_div, min_, max_, particals):
        self.curr_archiving_in = curr_archiving_in  # 当前存档中所有粒子的坐标
        self.curr_archiving_fit = curr_archiving_fit  # 当前存档中所有粒子的坐标
        self.mesh_div = mesh_div  # 等分因子，默认值为10等分，是个啥
        self.num_ = self.curr_archiving_in.shape[0]  # 当前存档中所有粒子的数量
        self.particals = particals  #这到底是个啥
        self.id_archiving = np.zeros((self.num_))  # 各个粒子的id编号，检索位与curr_archiving的检索位为相对应
        self.crowd_archiving = np.zeros((self.num_))  # 拥挤度矩阵，用于记录当前粒子所在网格的总粒子数，检索位与curr_archiving的检索为相对应
        self.probability_archiving = np.zeros((self.num_))  # 各个粒子被选为gbest的概率，检索位与curr_archiving的检索位为相对应
        self.gbest_in = np.zeros((self.particals, self.curr_archiving_in.shape[1]))  # 初始化gbest矩阵_坐标
        self.gbest_fit = np.zeros((self.particals, self.curr_archiving_fit.shape[1]))  # 初始化gbest矩阵_适应值
        self.min_ = min_
        self.max_ = max_

    def cal_mesh_id(self, in_):
        # 计算网格编号id
        # 首先，将每个维度按照等分因子进行等分离散化，获取粒子在各维度上的编号。按照10进制将每一个维度编号等比相加（如过用户自定义了mesh_div_num的值，则按照自定义），计算出值
        id_ = 0
        for i in range(self.curr_archiving_in.shape[1]):
            id_dim = int((in_[i] - self.min_[i]) * self.num_ / (self.max_[i] - self.min_[i]))
            id_ = id_ + id_dim * (self.mesh_div ** i)
        return id_

    def divide_archiving(self):
        # 进行网格划分，为每个粒子定义网格编号,什么东西
        for i in range(self.num_):
            self.id_archiving[i] = self.cal_mesh_id(self.curr_archiving_in[i])

    def get_crowd(self):#index_为[0, 0],应该改为7个元素的列表
        index_ = (np.linspace(0, self.num_ - 1, self.num_)).tolist()  # 定义一个数组存放粒子集的索引号，用于辅助计算
        index_ = list(map(int, index_))
        while (len(index_) > 0):
            index_same = [index_[0]]  # 存放本次子循环中与index[0]粒子具有相同网格id所有检索位
            for i in range(1, len(index_)):
                if self.id_archiving[index_[0]] == self.id_archiving[index_[i]]:
                    index_same.append(index_[i])
            number_ = len(index_same)  # 本轮网格中的总粒子数
            for i in index_same:  # 更新本轮网格id下的所有粒子的拥挤度
                self.crowd_archiving[i] = number_
                index_.remove(i)  # 删除本轮网格所包含的粒子对应的索引号，避免重复计算


class get_gbest(mesh_crowd):#一个继承mesh_crowd的类，放进去的是什么，出来的是全局最优的位置和目标函数值
    def __init__(self, curr_archiving_in, curr_archiving_fit, mesh_div_num, min_, max_, particals):
        super(get_gbest, self).__init__(curr_archiving_in, curr_archiving_fit, mesh_div_num, min_, max_, particals)
        self.divide_archiving()
        self.get_crowd()

        #print(mesh_div_num)
        #print(min_)
        #print(max_)
        #print(particals)

    def get_probability(self):
        for i in range(self.num_):
            self.probability_archiving = 10.0 / (self.crowd_archiving ** 3)
        self.probability_archiving = self.probability_archiving / np.sum(self.probability_archiving)  # 所有粒子的被选概率的总和为1

    def get_gbest_index(self):
        random_pro = random.uniform(0.0, 1.0)  # 生成一个0到1之间的随机数
        for i in range(self.num_):
            if random_pro <= np.sum(self.probability_archiving[0:i + 1]):
                return i  # 返回检索值

    def get_gbest(self):
        self.get_probability()
        for i in range(self.particals):
            gbest_index = self.get_gbest_index()
            self.gbest_in[i] = self.curr_archiving_in[gbest_index]  # gbest矩阵_坐标
            self.gbest_fit[i] = self.curr_archiving_fit[gbest_index]  # gbest矩阵_适应值
        return self.gbest_in, self.gbest_fit


class clear_archiving(mesh_crowd):
    def __init__(self, curr_archiving_in, curr_archiving_fit, mesh_div_num, min_, max_, particals):
        super(get_gbest, self).__init__(curr_archiving_in, curr_archiving_fit, mesh_div_num, min_, max_)
        self.divide_archiving()
        self.get_crowd()

    def get_probability(self):
        for i in range(self.num_):
            self.probability_archiving = self.crowd_archiving ** 2

    def get_clear_index(self):  # 按概率清除粒子，拥挤度高的粒子被清除的概率越高
        len_clear = (self.curr_archiving_in).shape[0] - self.thresh  # 需要清除掉的粒子数量
        clear_index = []
        while (len(clear_index) < len_clear):
            random_pro = random.uniform(0.0, np.sum(self.probability_archiving))  # 生成一个0到1之间的随机数
            for i in range(self.num_):
                if random_pro <= np.sum(self.probability_archiving[0:i + 1]):
                    if i not in clear_index:
                        clear_index.append(i)  # 记录检索值
        return clear_index

    def clear_(self, thresh):
        self.thresh = thresh
        self.archiving_size = archiving_size
        self.get_probability()
        clear_index = get_clear_index()
        self.curr_archiving_in = np.delete(self.curr_archiving_in[gbest_index], clear_index, axis=0)  # 初始化gbest矩阵_坐标
        self.curr_archiving_fit = np.delete(self.curr_archiving_fit[gbest_index], clear_index, axis=0)  # 初始化gbest矩阵_适应值
        return self.curr_archiving_in, self.curr_archiving_fit
