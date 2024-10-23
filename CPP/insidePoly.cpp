#include <iostream>
using namespace std;

struct Point {
	double	m_x;
	double	m_y;
};


bool pointInPolygon( Point point, Point* vs, int length )
{
	/*
	 * ray-casting algorithm based on
	 * http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
	 */

	double	x	= point.m_x;
	double	y	= point.m_y;
	cout<<"point(,"<<x<<","<<y<<")"<<endl;

	bool inside = false;
	for ( int i = 0, j = length - 1; i < length; j = i++ )
	{
		double	xi	= vs[j].m_x, yi = vs[j].m_y;
		double	xj	= vs[i].m_x, yj = vs[i].m_y;

         // 判断点是否位于多边形边的左右两侧
		bool intersect = ( (yi > y) != (yj > y) )
				 && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
		if ( intersect )
		{
			inside = !inside; // 每次交点都切换 inside 状态
		}
	}
	return(inside);
}


int main()
{
	Point vs[] = {
		{
			112.34619140625,
			31.372399104880525
		},
		{
			114.158935546875,
			29.19053283229458
		},
		{
			118.223876953125,
			28.815799886487298
		},
		{
			118.32275390624999,
			31.175209828310845
		},
		{
			115.72998046875,
			30.770159115784214
		},
		{
			114.3017578125,
			31.13760327002129
		},
		{
			115.57617187499999,
			31.98944183792288
		},
		{
			117.476806640625,
			31.475524020001806
		},
		{
			115.90576171874999,
			32.76880048488168
		},
		{
			112.34619140625,
			31.372399104880525
		}
	};
	Point p[] = {
		/*外*/
		{
			115.71899414062499,
			31.13760327002129
		},
		{
			116.45507812500001,
			29.99300228455108
		},
		/*外*/
		{
			116.54296874999999,
			32.69486597787505
		},
		{
			112.412109375,
			31.344254455668054
		},
		/*外*/
		{
			118.27880859375001,
			28.806173508854776
		}
	};
	for(auto pt:p){
		cout<<(pointInPolygon(pt,vs,10)?"在":"不在")<<endl;
	}
	return(0);
}