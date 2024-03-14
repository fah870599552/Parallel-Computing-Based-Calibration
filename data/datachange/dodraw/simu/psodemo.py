import numpy as np
import pandas as pd
from collections import Iterable, Counter
class PSO:
    def __init__(self, func, bound, POP_SIZE, w=1, c1=0.2, c2=0.2, v_max=0.05, *, var_name=None):
        """
        :param func: 传递函数引用，def or lambad
        :param bound: [(0, 60), (20, 70)]
        :param POP_SIZE: 粒子个数
        :param w:
        :param c1:
        :param c2:
        :param v_max: 初始速度最大值
        :param var_name:变量名，None为默认，前面加上*参数的意思是：想要给var_name赋值，必须用var_name=...的形式
        """
        bounds = Counter([isinstance(a, Iterable) for a in bound])[True]#看变量中有多少个变量是在范围内取值
        Var_size = int(np.ceil(POP_SIZE ** (1 / bounds)))
        vals = [np.linspace(var[0], var[1], Var_size) if isinstance(var, Iterable) else np.array([var]) for var in
                bound]#变量实现取值
        vals = np.array(list(map(lambda var: var.flatten(), np.meshgrid(*vals))))
        self.var_quantity, self.POP_SIZE = vals.shape
        self.func = func
        self.bound = bound
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.var_name = var_name
        #粒子
        self.particles = np.array(list(zip(*vals)))
        self.velocity = np.random.rand(*self.particles.shape) * v_max
        #粒子最优位置
        self.person_best = self.particles.copy()
        #全局最优位置
        self.global_best = max(self.person_best, key=lambda particle: self.func(*particle)).copy()

    def get_fitness(self):
        #获得所有粒子的适应度
        return np.array([self.func(*particle) for particle in self.particles])
        #这样能调用numpy中数组的矩阵运算，但这个函数中存在判断，因此不能简单送入方法，而选择用列表推导实现

    def update_position(self):
        for index, particle in enumerate(self.particles):
            V_k_plus_1 = self.w * self.velocity[index] \
                         + self.c1 * np.random.rand() * (self.person_best[index] - particle) \
                         + self.c2 * np.random.rand() * (self.global_best - particle)

            self.particles[index] = self.particles[index] + V_k_plus_1
            self.velocity[index] = V_k_plus_1

            for i, var in enumerate(particle):
                if isinstance(self.bound[i], Iterable):
                    if var < self.bound[i][0]:
                        self.particles[index][i] = self.bound[i][0]
                    elif var > self.bound[i][1]:
                        self.particles[index][i] = self.bound[i][1]
                elif var != self.bound[i]:
                    self.particles[index][i] = self.bound[i]
    def update_best(self):
        global_best_fitness = self.func(*self.global_best)
        person_best_value = np.array([self.func(*particle) for particle in self.person_best])

        for index, particle in enumerate(self.particles):
            current_particle_fitness = self.func(*particle)

            if current_particle_fitness > person_best_value[index]:
                person_best_value[index] = current_particle_fitness
                self.person_best[index] = particle
            if current_particle_fitness > global_best_fitness:
                global_best_fitness = current_particle_fitness
                self.global_best = particle
    def pso(self):
        self.update_position()
        self.update_best()
    def info(self):
        #输出当前粒子信息
        result = pd.DataFrame(self.particles)
        if self.var_name == None:
            result.columns = [f'x{i}' for i in range(len(self.bound))]
        else:
            result.columns = self.var_name
        result['fitness'] = self.get_fitness()
        return result

#评价函数

if __name__ == "__main__":
    func = lambda x, y, m: 30 * x - y if x < m and y < m else 30 * y - x if x < m and y >= m else x ** 2 - y / 2 if x >= m and y < m else 20 * (
                y ** 2) - 500 * x
    bound = ((0, 60), (0, 60), 30)
    var_name = ['x', 'y', 'm']
    POP_SIZE = 100
    w = 1
    c1 = 0.2
    c2 = 0.2
    v_max = 0.05
    pso = PSO(func, bound, POP_SIZE, w, c1, c2, v_max, var_name=var_name)
    for _ in range(1000):
        pso.pso()
        print(pso.get_fitness().sum())

    print(pso.global_best, func(*pso.global_best))
    print(pso.info())