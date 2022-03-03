import numpy as np
import math
import random
import os
from sklearn import metrics
import copy


# 樽海鞘算法，可以处理单目标和多目标
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
        # 多目标专用
        self.repository = []
        self.repository_size = 10

    # 帕累托操作符
    def pareto_dominance_operators(self, outside_object):
        # 0:features of object in repository < features of object outside
        # 1:features of object in repository > features of object outside
        repository = copy.deepcopy(self.repository)
        exchange_ = []

        for pareto_optimal_object in range(len(repository)):
            labels = [0 for i in range(len(self.measure_function))]
            for j in range(self.variable_num, self.variable_num + len(labels)):
                if repository[pareto_optimal_object][j] < outside_object[j]:
                    continue
                elif repository[pareto_optimal_object][j] == outside_object[j]:
                    labels[j - self.variable_num] = 2
                else:
                    labels[j - self.variable_num] = 1

            if labels == [1 for i in range(len(self.measure_function))]:
                exchange_.append(pareto_optimal_object)

            elif labels == [2 for i in range(len(self.measure_function))] or labels == [0 for i in range(len(self.measure_function))]:
                break

        if not exchange_:
            if len(self.repository) < self.repository_size:
                self.repository.append(outside_object)
            else:
                self.repository.pop(np.random.randint(0, len(self.repository)))
                self.repository.append(outside_object)
        else:

            for index in reversed(exchange_):
                del self.repository[index]
            self.repository.append(outside_object)

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

    def multi_combinatorial_model_optimization(self, variables_values):
        fitness_value = []
        ensemble_vectors = np.array([0.0 for i in range(len(self.result[0]))])
        for i in range(self.variable_num):
            ensemble_vectors += variables_values[i] * self.result[i]
        for j in self.measure_function:
            if j == "mse":
                fitness_value.append(metrics.mean_squared_error(np.array(self.result[-1]), ensemble_vectors))
            if j == 'mae':
                fitness_value.append(metrics.mean_absolute_error(np.array(self.result[-1]), ensemble_vectors))
            if j == 'mape':
                fitness_value.append(metrics.mean_absolute_percentage_error(np.array(self.result[-1]), ensemble_vectors))
            if j == 'std':
                fitness_value.append(np.std(np.array(self.result[-1]-ensemble_vectors)))
        return np.array(fitness_value)

    # Function: Initialize Variables
    def initial_position(self):
        position = np.zeros((self.swarm_size, len(self.min_values) + 1))
        for i in range(0, self.swarm_size):
            for j in range(0, len(self.min_values)):
                position[i, j] = random.uniform(self.min_values[j], self.max_values[j])
            # print(type(target_function))
            position[i, -1] = self.combinatorial_model_optimization(position[i, 0:position.shape[1] - 1])
        self.position = position

    def multi_initial_position(self):
        position = np.zeros((self.swarm_size, len(self.min_values) + len(self.measure_function)))
        for i in range(0, self.swarm_size):
            for j in range(0, len(self.min_values)):
                position[i, j] = random.uniform(self.min_values[j], self.max_values[j])
            position[i][self.variable_num:self.variable_num + len(self.measure_function)] = self.multi_combinatorial_model_optimization(position[i, 0:position.shape[1] - len(self.measure_function)])
        self.position = position

    # Function: Initialize Food Position
    def food_position(self, dimension):
        food = np.zeros((1, dimension + 1))
        for j in range(0, dimension):
            food[0, j] = 0.0
        food[0, -1] = self.combinatorial_model_optimization(food[0, 0:food.shape[1] - 1])
        self.food = food

    def multi_update_repository(self):
        for i in self.position:
            if not self.repository:
                self.repository.append(i)
            else:
                self.pareto_dominance_operators(i)

    # Function: Updtade Food Position by Fitness
    def update_food(self):
        for i in range(0, self.position.shape[0]):
            if self.food[0, -1] > self.position[i, -1]:
                for j in range(0, self.position.shape[1]):
                    self.food[0, j] = self.position[i, j]

    def multi_update_food(self):
        self.food = random.choice(self.repository)

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

    def multi_update_position(self, c1=1.0):
        for i in range(0, self.position.shape[0]):
            if i <= self.position.shape[0] / 2:
                for j in range(0, len(self.min_values)):
                    c2 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                    c3 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                    if c3 >= 0.5:  # c3 < 0.5

                        self.position[i, j] = np.clip(
                            (self.food[j] + c1 * ((self.max_values[j] - self.min_values[j]) * c2 + self.min_values[j])),
                            self.min_values[j],
                            self.max_values[j])
                    else:

                        self.position[i, j] = np.clip(
                            (self.food[j] - c1 * ((self.max_values[j] - self.min_values[j]) * c2 + self.min_values[j])),
                            self.min_values[j],
                            self.max_values[j])
            elif self.position.shape[0] / 2 < i < self.position.shape[0] + 1:
                for j in range(0, len(self.min_values)):
                    self.position[i, j] = np.clip(((self.position[i - 1, j] + self.position[i, j]) / 2),
                                                  self.min_values[j], self.max_values[j])
            self.position[i][
            self.variable_num:self.variable_num + len(self.measure_function)] = self.multi_combinatorial_model_optimization(
                self.position[i, 0:self.position.shape[1] - len(self.measure_function)])

    #SSA Function
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

    # MOSSA Function
    def multi_objective_salp_swarm_algorithm(self):
        count = 0
        self.multi_initial_position()
        self.multi_update_repository()
        # self.food_position(dimension=len(self.min_values))
        while count <= self.iterations:
            # print(self.repository)
            # print("Iteration = ", count, " f(x) = ", self.food[0, -1])
            c1 = 2 * math.exp(-(4 * (count / self.iterations)) ** 2)
            self.multi_update_food()
            self.multi_update_position(c1=c1)
            self.multi_update_repository()
            count = count + 1
        return self.food


#参数设置
number_of_models = 4  # 组合模型数量
swarm_size = 10  # 群体数，可自行设定
min_value_weights = [0.0 for i in range(number_of_models)]  # 每个模型的权重最小值，是一个列表
max_value_weights = [1.0 for j in range(number_of_models)]  # 每个模型的权重最大值，是一个列表
iterations = 1000  # 迭代次数


# 多目标优化调用
# 前5个参数不用动，后面的参数换成预测模型给出的结果向量，数据结构类型为np.ndarray, shape形式(n, ) 其中n为预测的结果数
# 输出为(模型数+指标数)维的向量，数据结构类型为np.ndarray，前面所有维度代表了每个模型的权重，最后x维是适应度函数(也就是mse，mae，mape)的值
loss_function = ["mape", 'std'] # 可选"mse", "mae", "mape", "std" 必须为list形式
opt_result = SalpSwarmAlgorithm(swarm_size, min_value_weights, max_value_weights, iterations, loss_function, mopso, moda, mogwo, mowoa, actual)  # 实例化优化方法
print(opt_result.multi_objective_salp_swarm_algorithm())

# 单目标优化调用
# 前5个参数不用动，后面的参数换成预测模型给出的结果向量，数据结构类型为np.ndarray, shape形式(n, ) 其中n为预测的结果数
# 输出为模型数+1维的向量，数据结构类型为np.ndarray，前面所有维度代表了每个模型的权重，最后一维是适应度函数(也就是mse，mae，mape)的值
loss_function = "mape"
opt_result = SalpSwarmAlgorithm(swarm_size, min_value_weights, max_value_weights, iterations, loss_function, mopso, moda, mogwo, mowoa, actual)  # 实例化优化方法
print(opt_result.salp_swarm_algorithm())
