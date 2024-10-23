#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

struct Point2d
{
    double x;
    double y;
    Point2d(double xx, double yy): x(xx), y(yy){}
};

// 计算多边形的质心
Point2d ComputeCentroid(const vector<Point2d> &points)
{
    double centroid_x = 0.0, centroid_y = 0.0;
    int point_num = points.size();
    for (const auto &point : points) {
        centroid_x += point.x;
        centroid_y += point.y;
    }
    centroid_x /= point_num;
    centroid_y /= point_num;
    return Point2d(centroid_x, centroid_y);
}

// 按极角排序顶点
vector<Point2d> SortPointsByAngle(const vector<Point2d> &points)
{
    Point2d centroid = ComputeCentroid(points);
    
    // 使用 lambda 表达式根据极角排序
    vector<Point2d> sorted_points = points;
    sort(sorted_points.begin(), sorted_points.end(), [&centroid](const Point2d &a, const Point2d &b) {
        double angle_a = atan2(a.y - centroid.y, a.x - centroid.x);
        double angle_b = atan2(b.y - centroid.y, b.x - centroid.x);
        return angle_a < angle_b;  // 按逆时针顺序排列
    });
    
    return sorted_points;
}

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

int main()
{
    // 输入乱序的多边形顶点
    vector<Point2d> points = { {0, 0}, {4, 0}, {4, 3}, {2, 5}, {0, 3} };
    
    // 先排序顶点
    vector<Point2d> sorted_points = SortPointsByAngle(points);
    
    // 计算面积
    double area = ComputePolygonArea(sorted_points);
    
    // 输出结果
    printf("Polygon area: %f\n", area);
    
    return 0;
}