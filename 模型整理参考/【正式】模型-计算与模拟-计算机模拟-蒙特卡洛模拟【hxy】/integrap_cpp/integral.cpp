#include <iostream>
#include <cstdlib>
#include <iomanip>
#include "monte_carlo.h"
using namespace std;

struct Point{ double x, y; };
struct pi_gen {
    Point operator () () {
        return (Point){double(rand() % 1000) / 1000.0, double(rand() % 1000) / 1000.0};
    } // 积分区间[0,1]
};

struct judge {
    bool operator () (const Point &x) {
        return x.y < x.x*x.x;
    } // 对y = x^2积分
};

Monte_carlo<Point, 10000000, pi_gen, judge> sim;

int main()
{
    cout << setprecision(9);
    cout << sim.getprob() << endl;
    return 0;
}