import numpy as np
import math
import random
import os
from sklearn import metrics


# 类封装的樽海鞘算法，单目标
class SalpSwarmAlgorithm():
    def __init__(self, swarm_size, min_values, max_values, iterations, measure_function, *result_vec):
        self.swarm_size = swarm_size
        self.variable_num = len(min_values)
        self.min_values = min_values
        self.max_values = max_values
        self.iterations = iterations
        self.measure_function = measure_function
        self.result = [i for i in result_vec]
        self.food = 0
        self.position =0

    def combinatorial_model_optimization(self, variables_values):
        ensemble_vectors = np.array([0.0 for i in range(len(self.result[0]))])
        for i in range(self.variable_num):
            ensemble_vectors += variables_values[i] * self.result[i]
        # print(ensemble_vectors)
        # print(result_vectors[-1])
        if self.measure_function == "mse":
            fitness_value = metrics.mean_squared_error(np.array(self.result[-1]), ensemble_vectors)
        if self.measure_function == 'mae':
            fitness_value = metrics.mean_absolute_error(np.array(self.result[-1]), ensemble_vectors)
        if self.measure_function == 'mape':
            fitness_value = metrics.mean_absolute_percentage_error(np.array(self.result[-1]), ensemble_vectors)
        return fitness_value

    # Function: Initialize Variables
    def initial_position(self):
        position = np.zeros((self.swarm_size, len(self.min_values) + 1))
        for i in range(0, self.swarm_size):
            for j in range(0, len(self.min_values)):
                position[i, j] = random.uniform(self.min_values[j], self.max_values[j])
            # print(type(target_function))
            position[i, -1] = self.combinatorial_model_optimization(position[i, 0:position.shape[1] - 1])
        self.position = position

    # Function: Initialize Food Position
    def food_position(self, dimension):
        food = np.zeros((1, dimension + 1))
        for j in range(0, dimension):
            food[0, j] = 0.0
        food[0, -1] = self.combinatorial_model_optimization(food[0, 0:food.shape[1] - 1])
        self.food = food

    # Function: Updtade Food Position by Fitness
    def update_food(self):
        for i in range(0, self.position.shape[0]):
            if self.food[0, -1] > self.position[i, -1]:
                for j in range(0, self.position.shape[1]):
                    self.food[0, j] = self.position[i, j]

    # Function: Updtade Position
    def update_position(self, c1=1.0):
        for i in range(0, self.position.shape[0]):
            if i <= self.position.shape[0] / 2:
                for j in range(0, len(self.min_values)):
                    c2 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                    c3 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                    if c3 >= 0.5:  # c3 < 0.5
                        self.position[i, j] = np.clip(
                            (self.food[0, j] + c1 * ((self.max_values[j] - self.min_values[j]) * c2 + self.min_values[j])), self.min_values[j],
                            self.max_values[j])
                    else:
                        self.position[i, j] = np.clip(
                            (self.food[0, j] - c1 * ((self.max_values[j] - self.min_values[j]) * c2 + self.min_values[j])), self.min_values[j],
                            self.max_values[j])
            elif self.position.shape[0] / 2 < i < self.position.shape[0] + 1:
                for j in range(0, len(self.min_values)):
                    self.position[i, j] = np.clip(((self.position[i - 1, j] + self.position[i, j]) / 2), self.min_values[j], self.max_values[j])
            self.position[i, -1] = self.combinatorial_model_optimization(self.position[i, 0:self.position.shape[1] - 1])

    # SSA Function
    def salp_swarm_algorithm(self):
        count = 0
        self.initial_position()
        self.food_position(dimension=len(self.min_values))
        while count <= self.iterations:
            print("Iteration = ", count, " f(x) = ", self.food[0,-1])
            c1 = 2*math.exp(-(4*(count/self.iterations))**2)
            self.update_food()
            self.update_position(c1=c1)
            count = count + 1
        return self.food[0]

#参数设置
loss_function = "mape"  # 可选"mse", "mae", "mape"
number_of_models = 4  # 组合模型数量
swarm_size = 5  # 群体数，可自行设定
min_value_weights = [0.0 for i in range(number_of_models)]  # 每个模型的权重最小值，是一个列表
max_value_weights = [1.0 for j in range(number_of_models)]  # 每个模型的权重最大值，是一个列表
iterations = 1000  # 迭代次数

# 前5个参数不用动，后面的参数换成预测模型给出的结果向量，数据结构类型为np.ndarray, shape形式(n, ) 其中n为预测的结果数
# 输出为模型数+1维的向量，数据结构类型为np.ndarray，前面所有维度代表了每个模型的权重，最后一维是适应度函数(也就是mse，mae，mape)的值
opt_result = SalpSwarmAlgorithm(swarm_size, min_value_weights, max_value_weights, iterations, loss_function, mopso, moda, mogwo, mowoa, actual)
print(opt_result.salp_swarm_algorithm())

