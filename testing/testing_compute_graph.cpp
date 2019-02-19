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

    tensor<float> *t0 = new tensor<float> ({5, 5}, {CONSTANT, {4}}, HOST);
    tensor<float> *t1 = new tensor<float> ({5, 5}, {CONSTANT, {5}}, HOST);


	variable<float> *v0 = new variable<float> ("t0", t0);
	variable<float> *v1 = new variable<float> ("t1", t1);

	auto sum = add_nocopy<float> (v0, v1);

	tensor<float> *fin = sum.eval();

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			assert( fin->get({i,j}) == 9 );
		}
	}


	delete t0;
	delete t1;
	delete v0;
	delete v1;
    return 0;
}
