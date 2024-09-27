#include <iostream>
#include "LeastSquareMethod.h"

/**
 * @brief Fit polynomial using Least Square Method.
 *
 * @param X X-axis coordinate vector of sample data.
 * @param Y Y-axis coordinate vector of sample data.
 * @param orders Fitting order which should be larger than zero.
 * @return Eigen::VectorXf Coefficients vector of fitted polynomial.
 */

using namespace std;

Eigen::VectorXf FitterLeastSquareMethod(vector<float> &X, vector<float> &Y, uint8_t orders)
{
    // abnormal input verification
    if (X.size() < 2 || Y.size() < 2 || X.size() != Y.size() || orders < 1)
        exit(EXIT_FAILURE);

    // map sample data from STL vector to eigen vector
    Eigen::Map<Eigen::VectorXf> sampleX(X.data(), X.size());
    Eigen::Map<Eigen::VectorXf> sampleY(Y.data(), Y.size());

    Eigen::MatrixXf mtxVandermonde(X.size(), orders + 1);  // Vandermonde matrix of X-axis coordinate vector of sample data
    Eigen::VectorXf colVandermonde = sampleX;              // Vandermonde column

    // construct Vandermonde matrix column by column
    for (size_t i = 0; i < orders + 1; ++i)
    {
        if (0 == i)
        {
            mtxVandermonde.col(0) = Eigen::VectorXf::Constant(X.size(), 1, 1);
            continue;
        }
        if (1 == i)
        {
            mtxVandermonde.col(1) = colVandermonde;
            continue;
        }
        colVandermonde = colVandermonde.array()*sampleX.array();
        mtxVandermonde.col(i) = colVandermonde;
    }

    // calculate coefficients vector of fitted polynomial
    Eigen::VectorXf result = (mtxVandermonde.transpose()*mtxVandermonde).inverse()*(mtxVandermonde.transpose())*sampleY;

    return result;
}

int main()
{
    float x[5] = {1, 2, 3, 4, 5};
    float y[5] = {7, 35, 103, 229, 431};

    vector<float> X(x, x + sizeof(x) / sizeof(float));
    vector<float> Y(y, y + sizeof(y) / sizeof(float));

    Eigen::VectorXf result(FitterLeastSquareMethod(X, Y, 3));

    cout << "\nThe coefficients vector is: \n" << endl;
    for (size_t i = 0; i < result.size(); ++i)
        cout << "theta_" << i << ": " << result[i] << endl;

    return 0;
}