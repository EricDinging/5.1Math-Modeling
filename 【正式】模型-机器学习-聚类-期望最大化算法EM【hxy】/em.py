import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import math

# 建立数据集
observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                         [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                         [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

# 定义一次EM步
def em_single(observations, theta):
    # 计算每轮次数
    length = [0 for i in range(5)]
    for i in range(5):
        length[i] = len(observations[i])
    # 计算每轮H的个数和T的个数
    num_H = [0 for i in range(5)]
    num_T = [0 for i in range(5)]
    for i in range(5):
      num_H[i] = observations[i].sum()
      num_T[i] = length[i] - observations[i].sum()
    # E步
    # 计算PA
    old_theta_a = theta[0]
    pro_A = [0 for i in range(5)]
    for i in range(5):
        pro_A[i] = binom.pmf(num_H[i],length[i],old_theta_a)
    # 计算PB
    old_theta_b = theta[1]
    pro_B = [0 for i in range(5)]
    for i in range(5):
        pro_B[i] = binom.pmf(num_H[i],length[i],old_theta_b)
    # 计算硬币A的概率
    PA = [0 for i in range(5)]
    for i in range(5):
        PA[i] = pro_A[i] / (pro_A[i] + pro_B[i])
    # 计算硬币B的概率
    PB = [0 for i in range(5)]
    for i in range(5):
        PB[i] = pro_B[i] / (pro_A[i] + pro_B[i])
    # 计算硬币A的H的期望，T的期望 
    E_A_H = 0
    E_A_T = 0
    for i in range(5):
      E_A_H += num_H[i] * PA[i]
      E_A_T += num_T[i] * PA[i]
    # 计算硬币B的H的期望，T的期望
    E_B_H = 0
    E_B_T = 0
    for i in range(5):
      E_B_H += num_H[i] * PB[i]
      E_B_T += num_T[i] * PB[i]
    # M步
    # 重新计算
    new_theta_A = E_A_H / (E_A_H + E_A_T)
    new_theta_B = E_B_H / (E_B_H + E_B_T)
    return [new_theta_A, new_theta_B]

# EM主函数
def em(ovservations, theta, tol=1e-6, iterations=10000):
    iteration = 0
    while iteration < iterations:
        new_theta = em_single(observations, theta)
        delta = np.abs(theta[0] - new_theta[0])
        if delta < tol:
            break;
        else:
            theta = new_theta
            iteration += 1
    return [new_theta, iteration]

# 打印结果
print(em(observations, [0.6, 0.5]))
