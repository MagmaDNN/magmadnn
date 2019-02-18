/**
 * @file testing_compute_graph.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#include "skepsi.h"

using namespace skepsi;

int main(int argc, char **argv) {

    tensor<float> *t0 = new tensor<float> ({5, 5}, {CONSTANT, {0}}, HOST);
    tensor<float> *t1 = new tensor<float> ({6, 4}, {CONSTANT, {0}}, HOST);


	variable<float> *v0 = new variable<float> ("t0", t0);
	variable<float> *v1 = new variable<float> ("t1", t1);

	
	printf("t0.size = %d\n", v0->eval()->get_size());
	printf("t1.size = %d\n", v1->eval()->get_size());


	delete t0;
	delete t1;
	delete v0;
	delete v1;
    return 0;
}
