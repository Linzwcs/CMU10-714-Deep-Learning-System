#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <assert.h>
namespace py = pybind11;
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
void matrix_transpose(const float *X, float *XT, int m, int n)
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
        X[x_start + i] = Y[y_start + i];
}

void arrcpy(unsigned char *X, const unsigned char *Y, int x_start, int y_start, int copysize)
{
    for (int i = 0; i < copysize; i++)
        X[x_start + i] = Y[y_start + i];
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
        }
        for (int j = 0; j < k; j++)
        {
            X[i * k + j] /= row_sum;
        }
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
                Iy[start + j] = 1;
            }
            else
            {
                Iy[start + j] = 0;
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t i = 0; i < m; i += batch)
    {
        /// init processs start
        int start = i;
        int end = std::min(m, i + batch);
        int mini_batch = end - start;

        float *batchX = new float[mini_batch * n];
        unsigned char *batchY = new unsigned char[mini_batch];
        float *batchIy = new float[mini_batch * k];
        float *batchXT = new float[mini_batch * n];
        float *batchZ = new float[mini_batch * k];
        float *batchZ_sub_Iy = new float[mini_batch * k];
        float *batch_grad = new float[n * k];

        arrcpy(batchX, X, 0, start * n, mini_batch * n);
        arrcpy(batchY, y, 0, start, mini_batch);
        onehot_matrix(batchY, batchIy, mini_batch, k);
        /// init processs end

        matrix_transpose(batchX, batchXT, mini_batch, n);
        matrix_dot(batchX, theta, batchZ, mini_batch, n, k);
        exp_normalize(batchZ, mini_batch, k);
        matrix_sub(batchZ, batchIy, batchZ_sub_Iy, mini_batch, k);
        matrix_dot(batchXT, batchZ_sub_Iy, batch_grad, n, mini_batch, k);
        matrix_div(batch_grad, n, k, mini_batch);
        matrix_mul(batch_grad, n, k, lr);
        matrix_sub(theta, batch_grad, theta, n, k);

        delete batchX;
        delete batchY;
        delete batchIy;
        delete batchXT;
        delete batchZ;
        delete batchZ_sub_Iy;
        delete batch_grad;
    }
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def("softmax_regression_epoch_cpp", [](py::array_t<float, py::array::c_style> X, py::array_t<unsigned char, py::array::c_style> y, py::array_t<float, py::array::c_style> theta, float lr, int batch)
          { softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch); }, py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"), py::arg("batch"));
}
