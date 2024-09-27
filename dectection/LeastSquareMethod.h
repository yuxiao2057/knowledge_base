#ifndef LEAST_SQUARE_METHOD_H
#define LEAST_SQUARE_METHOD_H
#include <Eigen/Dense>
#include <vector>

using namespace std;

/**
 * @brief Fit polynomial using Least Square Method.
 *
 * @param X X-axis coordinate vector of sample data.
 * @param Y Y-axis coordinate vector of sample data.
 * @param orders Fitting order which should be larger than zero.
 * @return Eigen::VectorXf Coefficients vector of fitted polynomial.
 */
Eigen::VectorXf FitterLeastSquareMethod(vector<float> &X, vector<float> &Y, uint8_t orders);

#endif