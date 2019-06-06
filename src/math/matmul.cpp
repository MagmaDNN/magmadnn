/**
 * @file matmul.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-06
 * 
 * @copyright Copyright (c) 2019
 */
#include "math/matmul.h"

namespace magmadnn {
namespace math {

template void matmul(int alpha, bool trans_A, Tensor<int> *A, bool trans_B, Tensor<int> *B, int beta, Tensor<int> *C) {

}

template void matmul(float alpha, bool trans_A, Tensor<float> *A, bool trans_B, Tensor<float> *B, float beta, Tensor<float> *C) {

}

template void matmul(double alpha, bool trans_A, Tensor<double> *A, bool trans_B, Tensor<double> *B, double beta, Tensor<double> *C) {

}

}
}