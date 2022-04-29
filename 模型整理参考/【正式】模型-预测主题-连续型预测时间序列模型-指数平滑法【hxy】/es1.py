import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def alpha_analysis(data,itype=2):
    '''
    判断误差最小的平滑系数
    :param data:   原始序列：list
    :param itype:  平滑类型：1,2,3
    :return:       返回平均绝对误差最小的平滑系数和最小平均绝对误差
    '''
    alpha_all = [0.01 * i for i in range(1,100)]  #只需要0.1-0.9修改为alpha_triple = [0.1 * i for i in range(1,10)]
    best_alpha = 0
    min_MAE = float('Inf') #  无穷大
    if itype == 2:
        for i in range(len(alpha_all)):
            alpha = alpha_all[i]
            a_double,b_double,F_double = exponential_smoothing_2(alpha, data)
            AE_double, MAE_double, RE_double, MRE_double = model_error_analysis(F_double, data)
            if MAE_double <= min_MAE:
                min_MAE = MAE_double
                best_alpha = alpha
            else:
                pass
    elif itype == 3:
        for i in range(len(alpha_all)):
            alpha = alpha_all[i]
            a_triple, b_triple, c_triple, F_triple = exponential_smoothing_3(alpha, data)
            AE_triple, MAE_triple, RE_triple, MRE_triple = model_error_analysis(F_triple, data)
            if MAE_triple <= min_MAE:
                min_MAE = MAE_triple
                best_alpha = alpha
            else:
                pass
    else:
        for i in range(len(alpha_all)):
            alpha = alpha_all[i]
            F_single = exponential_smoothing_1(alpha, data)
            AE_single, MAE_single, RE_single, MRE_single = model_error_analysis(F_single, data)
            if MAE_single <= min_MAE:
                min_MAE = MAE_single
                best_alpha = alpha
            else:
                pass
    
    return best_alpha, min_MAE

def model_error_analysis(F, data):
    '''
    误差分析
    :param F:     预测数列：list
    :param data:  原始序列：list
    :return:      返回各期绝对误差，相对误差：list，返回平均绝对误差和平均相对误差
    '''
    AE = [0 for i in range(len(data)-1)]
    RE = []
    AE_num = 0
    RE_num = 0
    for i in range(1,len(data)):
        _AE = abs(F[i-1] - data[i])
        _RE = _AE / data[i]
        AE_num += _AE
        RE_num += _RE
        AE[i-1] = _AE
        RE.append('{:.2f}%'.format(_RE*100))
    MAE = AE_num / (len(data)-1)
    MRE = '{:.2f}%'.format(RE_num *100 / (len(data)-1))
    return AE, MAE, RE, MRE

def exponential_smoothing_1(alpha, data):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param data:   数据序列：list
    :return:       返回一次指数平滑值：list
    '''
    s_single=[]
    s_single.append((data[0]+data[1]+data[2])/3)
    for i in range(1, len(data)+1):
        s_single.append(alpha * data[i-1] + (1 - alpha) * s_single[i-1])
    return s_single

t = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
data = [10,15,8,20,10,16,18,20,22,24,20,26,27,29,29]
alpha_analysis(data,itype=1)
alpha = 0.5
s = exponential_smoothing_1(alpha, data)
print(s)

