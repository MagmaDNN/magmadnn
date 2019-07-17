#include "magmadnn.h"
#include "utilities.h"

using namespace magmadnn;

void test_indexing(memory_t mem, bool verbose);
void test_fill(tensor_filler_t<float> filler, memory_t mem, bool verbose);
void test_copy(memory_t mem, bool verbose);
void test_big_4(memory_t mem, unsigned int size);

int main(int argc, char **argv) {
    magmadnn_init();

    // test indexing
    test_indexing(HOST, true);
#if defined(_HAS_CUDA_)
    test_indexing(DEVICE, true);
    test_indexing(MANAGED, true);
    test_indexing(CUDA_MANAGED, true);
#endif

    test_fill({CONSTANT, {0.5}}, HOST, true);
#if defined(_HAS_CUDA_)
    test_fill({CONSTANT, {0.5}}, DEVICE, true);
    test_fill({CONSTANT, {0.5}}, MANAGED, true);
    test_fill({CONSTANT, {0.5}}, CUDA_MANAGED, true);
#endif

    // test copying
    test_copy(HOST, true);
#if defined(_HAS_CUDA_)
    test_copy(DEVICE, true);
    test_copy(MANAGED, true);
    test_copy(CUDA_MANAGED, true);
#endif

    test_for_all_mem_types(test_big_4, 10);

    magmadnn_finalize();
    return 0;
}

void test_indexing(memory_t mem, bool verbose) {
    unsigned int x_size = 10, y_size = 28, z_size = 28;

    if (verbose) printf("Testing indexing on device %s...  ", get_memory_type_name(mem));

    Tensor<float> *t = new Tensor<float>({x_size, y_size, z_size}, mem);

    // test
    for (int i = 0; i < (int) x_size; i++)
        for (int j = 0; j < (int) y_size; j++)
            for (int k = 0; k < (int) z_size; k++) t->set({i, j, k}, i * j * k);

    for (int i = 0; i < (int) x_size; i++) {
        for (int j = 0; j < (int) y_size; j++) {
            for (int k = 0; k < (int) z_size; k++) {
                MAGMADNN_TEST_ASSERT_DEFAULT(t->get({i, j, k}) == i * j * k,
                                             "\"t->get({i, j, k}) == i * j * k\" failed");
            }
        }
    }

    /* try the [] indexing */
    for (unsigned int i = 0; i < x_size; i++) {
        for (unsigned int j = 0; j < y_size; j++) {
            for (unsigned int k = 0; k < z_size; k++) {
                float val = (*t)[{i, j, k}];
                MAGMADNN_TEST_ASSERT_DEFAULT(val == i * j * k, "\"val == i * j * k\" failed");
            }
        }
    }

    delete t;

    if (verbose) show_success();
}

void test_fill(tensor_filler_t<float> filler, memory_t mem, bool verbose) {
    unsigned int x_size = 50, y_size = 30;

    if (verbose) printf("Testing fill_constant on %s...  ", get_memory_type_name(mem));
    if (filler.values.size() == 0) {
        fprintf(stderr, "tester error.\n");
        return;
    }

    float val = filler.values[0];
    Tensor<float> *t = new Tensor<float>({x_size, y_size}, filler, mem);

    for (int i = 0; i < (int) x_size; i++) {
        for (int j = 0; j < (int) y_size; j++) {
            MAGMADNN_TEST_ASSERT_DEFAULT(t->get({i, j}) == val, "\"t->get({i, j}) == val\" failed");
        }
    }
    if (verbose) show_success();
}

void test_copy(memory_t mem, bool verbose) {
    unsigned int x_size = 10, y_size = 28, z_size = 28;
    unsigned int x_size_new = 4, y_size_new = 28, z_size_new = 1;

    if (verbose) printf("Testing copying on device %s...  ", get_memory_type_name(mem));

    Tensor<float> *t = new Tensor<float>({x_size, y_size, z_size}, mem);
    Tensor<float> *t_new = new Tensor<float>({x_size_new, y_size_new, z_size_new}, mem);

    // test
    for (unsigned int i = 0; i < x_size; i++) {
        for (unsigned int j = 0; j < y_size; j++) {
            for (unsigned int k = 0; k < z_size; k++) {
                t->set({i, j, k}, i * j * k);
            }
        }
    }

    t_new->copy_from(*t, {x_size_new, y_size_new, z_size_new});

    for (unsigned int i = 0; i < x_size_new; i++) {
        for (unsigned int j = 0; j < y_size_new; j++) {
            for (unsigned int k = 0; k < z_size_new; k++) {
                MAGMADNN_TEST_ASSERT_DEFAULT(t_new->get({i, j, k}) == i * j * k,
                                             "\"t_new->get({i, j, k}) == i * j * k\" failed");
            }
        }
    }

    delete t;

    if (verbose) show_success();
}

void test_big_4(memory_t mem, unsigned int size) {
    printf("Testing big_4 on %s...  ", get_memory_type_name(mem));

    /* test the copy-constructor, move-constructor, destructor, and assignment operator of Tensor */

    Tensor<float> x({size, size}, {CONSTANT, {1.0f}}, mem);

    /* test assignment */
    Tensor<float> x_assigned = x;

    for (unsigned int i = 0; i < x_assigned.size(); i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(x_assigned.get(i), x.get(i));
    }

    show_success();
}
