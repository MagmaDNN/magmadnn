#include "skepsi.h"

using namespace skepsi;

const char* get_memory_type_name(memory_t mem);
void test_indexing(memory_t mem, bool verbose);
void test_fill(tensor_filler_t filler, memory_t mem, bool verbose);

int main(int argc, char **argv) {
    
	// test indexing	
    test_indexing(HOST, true);
	#ifdef _HAS_CUDA_
	test_indexing(DEVICE, true);
	test_indexing(MANAGED, true);
	test_indexing(CUDA_MANAGED, true);
	#endif

	test_fill({CONSTANT, {0.5}}, HOST, true);
	#ifdef _HAS_CUDA_
	test_fill({CONSTANT, {0.5}}, DEVICE, true);
	test_fill({CONSTANT, {0.5}}, MANAGED, true);
	test_fill({CONSTANT, {0.5}}, CUDA_MANAGED, true);
	#endif

    return 0;
}

void test_indexing(memory_t mem, bool verbose) {
	unsigned int x_size = 10, y_size = 28, z_size = 28;

	if (verbose) printf("Testing indexing on device %s...  ", get_memory_type_name(mem));
	
    tensor<float> *t = new tensor<float> ({x_size, y_size, z_size}, mem);

    // test
    for (int i = 0; i < (int)x_size; i++)
        for (int j = 0; j < (int)y_size; j++)
            for (int k = 0; k < (int)z_size; k++)
                t->set({i,j,k}, i*j*k);

    for (int i = 0; i < (int)x_size; i++)
        for (int j = 0; j < (int)y_size; j++)
            for (int k = 0; k < (int)z_size; k++)
				assert( t->get({i,j,k}) == i*j*k );
    
    delete t;

	if (verbose) printf("Success!\n");
}

void test_fill(tensor_filler_t filler, memory_t mem, bool verbose) {
	unsigned int x_size = 50, y_size = 30;

	if (verbose) printf("Testing fill_constant on %s...  ", get_memory_type_name(mem));
	if (filler.values.size() == 0) { fprintf(stderr, "tester error.\n"); return; }

	float val = filler.values[0];
	tensor<float> *t = new tensor<float> ({x_size, y_size}, filler, mem);

	for (int i = 0; i < (int) x_size; i++) {
		for (int j = 0; j < (int) y_size; j++) {
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

