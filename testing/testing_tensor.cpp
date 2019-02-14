#include "skepsi.h"

using namespace skepsi;

const char* get_memory_type_name(memory_t mem);
void test_indexing(memory_t mem, bool verbose);
void test_fill(memory_t mem, bool verbose);

int main(int argc, char **argv) {
    
	// test indexing	
    test_indexing(HOST, true);
	#ifdef _HAS_CUDA_
	test_indexing(DEVICE, true);
	test_indexing(MANAGED, true);
	test_indexing(CUDA_MANAGED, true);
	#endif

	test_fill(HOST, true);
	#ifdef _HAS_CUDA_
	test_fill(DEVICE, true);
	test_fill(MANAGED, true);
	test_fill(CUDA_MANAGED, true);
	#endif

    return 0;
}

void test_indexing(memory_t mem, bool verbose) {

	if (verbose) printf("Testing indexing on device %s...  ", get_memory_type_name(mem));
	
    tensor<float> *t = new tensor<float> ({4, 3, 2}, mem);

    // test
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 2; k++)
                t->set({i,j,k}, i*j*k);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 2; k++)
				assert( t->get({i,j,k}) == i*j*k );
    
    delete t;

	if (verbose) printf("Success!\n");
}

void test_fill(memory_t mem, bool verbose) {
	unsigned int x_size = 10, y_size = 5;

	if (verbose) printf("Testing fill_constant on %s...  ", get_memory_type_name(mem));
	float val = 0.5;
	tensor_filler_t filler = { CONSTANT, {val} };

	tensor<float> *t = new tensor<float> ({x_size, y_size}, filler, mem);

	for (int i = 0; i < (int) x_size; i++) {
		for (int j = 0; j < (int) y_size; j++) {
			printf("{%d, %d} == %.3f\n", i, j, t->get({i,j}));
			assert( t->get({i,j}) == val );
		}
	}
	if (verbose) printf("Success!\n");
}


const char* get_memory_type_name(memory_t mem) {
	switch (mem) {
		case HOST: 			return "HOST";
		#ifdef _HAS_CUDA_
		case DEVICE: 		return "DEVICE";
		case MANAGED: 		return "MANAGED";
		case CUDA_MANAGED: 	return "CUDA_MANAGED";
		#endif
		default: 			return "UNDEFINED_MEMORY_TYPE";
	}
}

