#include<vector>
#include<unordered_map>
#include<string>
using namespace std;

class Solution {
    int findf(vector<int>& father, vector<double>& weight, int x) {
        if (father[x] != x) {
            int fatherX = findf(father, weight, father[x]);
            weight[x] = weight[x] * weight[father[x]];
            father[x] = fatherX;
        }
        return father[x];
    }

    void merge(vector<int>& father, vector<double>& weight, int x, int y, double val) {
        int fatherX = findf(father, weight, x);
        int fatherY = findf(father, weight, y);
        father[fatherX] = fatherY;
        weight[fatherX] = val * weight[y] / weight[x];
    }

public:
    vector<double> calcEquation(vector<vector<string>>& equations,
                                vector<double>& values,
                                vector<vector<string>>& queries) {
        int nvars = 0;
        unordered_map<string, int> variables;

        int n = equations.size();
        for (int i = 0; i < n; i++) {
            if (variables.find(equations[i][0]) == variables.end()) {
                variables[equations[i][0]] = nvars++;
            }
            if (variables.find(equations[i][1]) == variables.end()) {
                variables[equations[i][1]] = nvars++;
            }
        }
        vector<int> f(nvars);
        vector<double> w(nvars, 1.0);
        for (int i = 0; i < nvars; i++) {
            f[i] = i;
        }

        for (int i = 0; i < n; i++) {
            merge(f, w, variables[equations[i][0]], variables[equations[i][1]], values[i]);
        }
        vector<double> ret;
        for (const auto& q : queries) {
            double result = -1.0;
            if (variables.find(q[0]) != variables.end() &&
                variables.find(q[1]) != variables.end()) {
                int ia = variables[q[0]], ib = variables[q[1]];
                int fa = findf(f, w, ia), fb = findf(f, w, ib);
                if (fa == fb) {
                    result = w[ia] / w[ib];
                }
            }
            ret.push_back(result);
        }
        return ret;
    }
};