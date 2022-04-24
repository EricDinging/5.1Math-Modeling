'''
求解pi
修改generate和judge两个函数对象即可
根据需要对输出的概率进行进一步加工
'''
import random
import math
class monte_carlo(object):
    def __init__(self, rand_times, gen, cond):
        self.rand_times = rand_times
        self.generator = gen
        self.condition = cond
    def getprob(self):
        cnt = 0
        for i in range(self.rand_times):
            x = self.generator()
            if (self.condition(x)):
                cnt = cnt + 1
        return 1.0 * cnt / self.rand_times


def generate():
    return [random.random(), random.random()]

def judge(x):
    return math.sqrt((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5)) < 0.5

if __name__ == "__main__":
    sim = monte_carlo(100000, generate, judge)
    print (sim.getprob() * 4.0)
