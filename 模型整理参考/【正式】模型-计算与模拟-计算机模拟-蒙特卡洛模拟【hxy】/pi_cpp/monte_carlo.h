#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

template<class Point, int rand_times, class gen, class cond>
class Monte_carlo {
/*
 * dimension 维度
 * rand_times 模拟次数
 * gen 生成点的方式
 * cond 计数条件
 */
private:
    gen generator;
    cond condition;

public:
    double getprob() {
        int cnt = 0;
        for (int i = 1; i <= rand_times; ++i) {
            Point x = generator();
            if (condition(x)) ++cnt;
        }
        return double(cnt) / rand_times;
    }
};

#endif //MONTE_CARLO_H