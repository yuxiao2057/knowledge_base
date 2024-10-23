#include <vector>
#include <cmath>

using namespace std;

struct Point2d
{
    double x;
    double y;
    Point2d(double xx, double yy): x(xx), y(yy){}
};
 
//计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列
double ComputePolygonArea(const vector<Point2d> &points)
{
    int point_num = points.size();
    if(point_num < 3)return 0.0;
    double s = 0;
    for(int i = 0; i < point_num; ++i)
        s += points[i].x * points[(i+1)%point_num].y - points[i].y * points[(i+1)%point_num].x;
    return fabs(s/2.0);
}