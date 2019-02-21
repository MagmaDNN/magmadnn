/**
 * @file tensor_math.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-21
 * 
 * @copyright Copyright (c) 2019
 */
#include <stdio.h>
#include "skepsi.h"

using namespace skepsi;

int main(int argc, char **argv) {

    /* AFFINE TRANSFORMATION 
        This example shows you how to create tensors A,x, and b. Then
        calculate the affine transformation Ax+b and print it out. */

    // A: MxN, x: Nx1, and b: Mx1
    const unsigned int M = 5; 
    const unsigned int N = 4;
    
    /* initialize the tensor with three parameters:
        shape: a vector of axis sizes
        filler: a struct dictating how to fill the tensor (this is optional)
        memory type: HOST means these tensors will be stored in RAM for the CPU
    */
    tensor<float> *A_tensor = new tensor<float> ({M, N}, {CONSTANT, {3}}, HOST);
    tensor<float> *x_tensor = new tensor<float> ({N, 1}, {CONSTANT, {2}}, HOST);
    tensor<float> *b_tensor = new tensor<float> ({M, 1}, {ONE, {}}, HOST);

    /* now we wrap the tensors in variables so that we can use them in the
        compute graph. All compute graph methods are in skepsi::op.
        op::var takes two arguments:
            name: a string name for the variable (functionality is not dependent on this)
            tensor: the tensor value that the variable points to
        op::var returns an allocated pointer (op::variable<float> *)
    */
    auto A = op::var("A", A_tensor);
    auto x = op::var("x", x_tensor);
    auto b = op::var("b", b_tensor);

    /* create the compute tree. nothing is evaluated. 
        returns an operation pointer (op::operation<float> *)
    */
    auto aff = op::add( op::matmul(A, x), b);

    /* get the final tensor result by evaluating the compute tree */
    tensor<float> *final_val = aff->eval();

    /* print out the results */
    printf("Ax+b = {");
    for (int i = 0; i < (int) M; i++) {
        printf("%.1f%s", final_val->get({i,0}), (i!=(int)M-1)?", ":"}\n");
    }

    /* free up allocated memory. free the head of the tree, will free 
        all of the nodes in the tree.
    */
    delete A_tensor;
    delete x_tensor;
    delete b_tensor;
    delete aff;

    return 0;
}