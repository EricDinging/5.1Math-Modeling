#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include "monte_carlo.h"
using namespace std;

struct Point{ double x, y; };
struct pi_gen {
    Point operator () () {
        return (Point){double(rand() % 1000) / 1000.0, double(rand() % 1000) / 1000.0};
    }
};
struct judge {
    bool operator () (const Point &x) {
        return sqrt((x.x-0.5)*(x.x-0.5)+(x.y-0.5)*(x.y-0.5)) < 0.5;
    }
};

Monte_carlo<Point, 10000000, pi_gen, judge> sim;

int main()
{
    cout << setprecision(9);
    cout << sim.getprob()*4 << endl;
    return 0;
}