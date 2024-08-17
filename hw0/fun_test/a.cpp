
#include <cmath>
#include <iostream>
#include <assert.h>

void matrix_dot(float *X, float *Y, float *Z, int m, int n, int p)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            Z[i * p + j] = 0;
            for (int k = 0; k < n; k++)
            {
                Z[i * p + j] += X[i * n + k] * Y[k * p + j];
            }
        }
    }
}

void matrix_transpose(float *X, float *XT, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            XT[j * m + i] = X[i * n + j];
        }
    }
}

void matrix_sub(float *X, float *Y, float *Z, int m, int n)
{
    for (int i = 0; i < m * n; i++)
        Z[i] = X[i] - Y[i];
}

void arrcpy(float *X, const float *Y, int x_start, int y_start, int copysize)
{
    for (int i = 0; i < copysize; i++)
    {
        X[x_start + i] = Y[y_start + i];
    }
}

void arrcpy(unsigned char *X, const unsigned char *Y, int x_start, int y_start, int copysize)
{
    for (int i = 0; i < copysize; i++)
    {
        X[x_start + i] = Y[y_start + i];
    }
}

void exp_normalize(float *X, int m, int k)
{
    for (int i = 0; i < m; i++)
    {
        float row_sum = 0;
        for (int j = 0; j < k; j++)
        {
            X[i * k + j] = exp(X[i * k + j]);
            row_sum += X[i * k + j];
            // std::cout << X[i * k + j] << " ";
        }
        for (int j = 0; j < k; j++)
        {
            X[i * k + j] /= row_sum;
        }
        // std::cout << "sum :" << row_sum << std::endl;
    }
}

void matrix_div(float *X, int m, int n, float div_value)
{
    for (int i = 0; i < m * n; i++)
        X[i] /= div_value;
}

void matrix_mul(float *X, int m, int n, float mul_value)
{
    for (int i = 0; i < m * n; i++)
        X[i] *= mul_value;
}

void onehot_matrix(unsigned char *y, float *Iy, int m, int k)
{
    for (int i = 0; i < m; i++)
    {
        int start = i * k;
        int idx = y[i];
        for (int j = 0; j < k; j++)
        {
            if (j == idx)
            {
                // std::cout << j << std::endl;
                Iy[start + j] = 1;
            }
            else
            {
                Iy[start + j] = 0;
            }
        }
    }
}

void assert_equal(const float *X, const float *Y, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        std::cout << X[i] << " " << Y[i] << std::endl;
        assert(fabs(X[i] - Y[i]) < 1e-5);
    }
}

int main()
{
    float Z[25];
    float X[25] = {0.63992102, 0.14335329, 0.94466892, 0.52184832, 0.41466194,
                   0.26455561, 0.77423369, 0.45615033, 0.56843395, 0.0187898,
                   0.6176355, 0.61209572, 0.616934, 0.94374808, 0.6818203,
                   0.3595079, 0.43703195, 0.6976312, 0.06022547, 0.66676672,
                   0.67063787, 0.21038256, 0.1289263, 0.31542835, 0.36371077};
    float Y[25] = {0.63992102, 0.26455561, 0.6176355, 0.3595079, 0.67063787,
                   0.14335329, 0.77423369, 0.61209572, 0.43703195, 0.21038256,
                   0.94466892, 0.45615033, 0.616934, 0.6976312, 0.1289263,
                   0.52184832, 0.56843395, 0.94374808, 0.06022547, 0.31542835,
                   0.41466194, 0.0187898, 0.6818203, 0.66676672, 0.36371077};
    float *XT = new float[25];
    matrix_transpose(X, XT, 5, 5);
    assert_equal(XT, Y, 25);

    std::cout << "<<<<<<<<<pass transpose<<<<<<<" << std::endl;

    float X_Y_dot[25] = {1.76671864, 1.0156224, 1.84100052, 1.25964848, 0.89652974,
                         1.0156224, 1.20097081, 1.46798843, 0.79846201, 0.58525029,
                         1.84100052, 1.46798843, 2.49228169, 1.43139539, 1.16819332,
                         1.25964848, 0.79846201, 1.43139539, 1.25513711, 0.68449358,
                         0.89652974, 0.58525029, 1.16819332, 0.68449358, 0.74241853};

    matrix_dot(X, Y, Z, 5, 5, 5);
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            std::cout << Z[i * 5 + j] << " ";
        }
        std::cout << std::endl;
    }
    assert_equal(Z, X_Y_dot, 25);

    std::cout << "<<<<<<<<<pass matrix dot<<<<<<<" << std::endl;

    float sub[25] = {0., -0.12120232, 0.32703342, 0.16234042, -0.25597593,
                     0.12120232, 0., -0.15594539, 0.131402, -0.19159276,
                     -0.32703342, 0.15594539, 0., 0.24611688, 0.552894,
                     -0.16234042, -0.131402, -0.24611688, 0., 0.35133836,
                     0.25597593, 0.19159276, -0.552894, -0.35133836, 0.};
    matrix_sub(X, Y, Z, 5, 5);
    assert_equal(Z, sub, 25);

    std::cout << "<<<<<<<<<pass matrix sub<<<<<<<" << std::endl;

    float copy[10];
    float true_copy[10] = {0.26455561, 0.77423369, 0.45615033, 0.56843395, 0.0187898,
                           0.6176355, 0.61209572, 0.616934, 0.94374808, 0.6818203};
    arrcpy(copy, X, 0, 5, 10);
    assert_equal(copy, true_copy, 10);
    std::cout << "<<<<<<<<<pass arr copy<<<<<<<" << std::endl;

    float normalized_matrix[25] = {0.21496871, 0.1308334, 0.29155841, 0.19102795, 0.17161153,
                                   0.16630235, 0.27685269, 0.201422, 0.225357, 0.13006596,
                                   0.18363331, 0.18261884, 0.18350454, 0.25443705, 0.19580626,
                                   0.17908423, 0.1935199, 0.25113222, 0.1327641, 0.24349955,
                                   0.27400226, 0.17292899, 0.15940128, 0.19208286, 0.20158462};
    arrcpy(Z, X, 0, 0, 25);
    exp_normalize(Z, 5, 5);
    assert_equal(Z, normalized_matrix, 25);
    std::cout << "<<<<<<<<<pass exp normalize<<<<<<<" << std::endl;
    float Iy[10];
    float TrueIy[10] = {1., 0., 0., 1., 1., 0., 0., 1., 1., 0.};
    unsigned char y[5] = {0, 1, 0, 1, 0};
    onehot_matrix(y, Iy, 5, 2);
    assert_equal(Iy, TrueIy, 10);
    std::cout << "<<<<<<<<<pass onehot<<<<<<<" << std::endl;

    std::cout << "<<<<<<<<<start test<<<<<<<" << std::endl;
    float testX[25] = {0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548,
                       0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152,
                       0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606,
                       0.0871293, 0.0202184, 0.83261985, 0.77815675, 0.87001215,
                       0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443};

    float theta[10] = {
        0.63992102, 0.14335329, 0.94466892, 0.52184832, 0.41466194,
        0.26455561, 0.77423369, 0.45615033, 0.56843395, 0.0187898};
    unsigned char testy[5] = {1, 1, 0, 0, 0};
    float testIy[10];
    float testZ[10];
    onehot_matrix(testy, testIy, 5, 2);
    matrix_dot(testX, theta, testZ, 5, 5, 2);

    float desired[10] = {1.93944418, 0.86786806, 2.16054193, 1.00364863, 1.99882595,
                         0.96332466, 1.51713094, 0.61461928, 2.24407981, 1.03767566};

    assert_equal(testZ, desired, 10);

    float exp_desired[10] = {0.74489654, 0.25510346, 0.76076775, 0.23923225, 0.73798105,
                             0.26201895, 0.71146538, 0.28853462, 0.76966209, 0.23033791};

    std::cout << "<<<<<<<<<<<<<<<" << std ::endl;
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            std::cout << testZ[i * 2 + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "<<<<<<<<<<<<<<<" << std ::endl;
    exp_normalize(testZ, 5, 2);
    std::cout << "<<<<<<<<<<<<<<<" << std ::endl;
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            std::cout << testZ[i * 2 + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "<<<<<<<<<<<<<<<" << std ::endl;
    assert_equal(testZ, exp_desired, 10);

    return 0;
}