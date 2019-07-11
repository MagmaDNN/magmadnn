/*
MAGMADNN_TEST_ASSERT_DEFAULT(mm->get(i) == i * val, "\"mm->get(i) == i * val\" failed");
    This is a test for the memorymanager class in magmadnn
*/

#include <assert.h>
#include <stdio.h>
#include "magmadnn.h"
#include "utilities.h"

using namespace magmadnn;

void test_get_set(memory_t mem, int size, bool verbose) {
    float val = 1.125f;

    if (verbose) printf("Testing %s get/set...  ", get_memory_type_name(mem));

    MemoryManager<float> *mm = new MemoryManager<float>(size, mem, (device_t) 0);

    // set
    for (int i = 0; i < (int) size; i++) {
        mm->set(i, i * val);
    }

    // test
    for (int i = 0; i < (int) size; i++) {
        MAGMADNN_TEST_ASSERT_DEFAULT(mm->get(i) == i * val, "\"mm->get(i) == i * val\" failed");
    }

    delete mm;

    if (verbose) show_success();
}

void test_copy(memory_t src_mem, memory_t dst_mem, int size, bool verbose) {
    if (verbose) printf("Testing %s->%s copy...  ", get_memory_type_name(src_mem), get_memory_type_name(dst_mem));

    // create
    MemoryManager<float> *mm_src = new MemoryManager<float>(size, src_mem, (device_t) 0);
    MemoryManager<float> *mm_dst = new MemoryManager<float>(size, dst_mem, (device_t) 0);

    // fill in mm_src
    for (int i = 0; i < size; i++) mm_src->set(i, 2 * i + 1);

    // copy mm_src into mm_dst
    mm_dst->copy_from(*mm_src);

    // test for success
    for (int i = 0; i < size; i++)
        MAGMADNN_TEST_ASSERT_DEFAULT(mm_src->get(i) == mm_src->get(i), "\"mm_src->get(i) == mm_src->get(i)\" failed");

    // free
    delete mm_src;
    delete mm_dst;

    if (verbose) show_success();
}

int main(int argc, char **argv) {
    magmadnn_init();

    unsigned int test_size = 100;

    if (argc == 2) test_size = atoi(argv[1]);

    // get/set
    test_get_set(HOST, test_size, true);
#if defined(_HAS_CUDA_)
    test_get_set(DEVICE, test_size, true);
    test_get_set(MANAGED, test_size, true);
    test_get_set(CUDA_MANAGED, test_size, true);
#endif

    // test copy
    // host to ...
    test_copy(HOST, HOST, test_size, true);

#if defined(_HAS_CUDA_)
    test_copy(HOST, DEVICE, test_size, true);
    test_copy(HOST, MANAGED, test_size, true);
    test_copy(HOST, CUDA_MANAGED, test_size, true);

    // device to ..
    test_copy(DEVICE, HOST, test_size, true);
    test_copy(DEVICE, DEVICE, test_size, true);
    test_copy(DEVICE, MANAGED, test_size, true);
    test_copy(DEVICE, CUDA_MANAGED, test_size, true);

    // managed to ..
    test_copy(MANAGED, HOST, test_size, true);
    test_copy(MANAGED, DEVICE, test_size, true);
    test_copy(MANAGED, MANAGED, test_size, true);
    test_copy(MANAGED, CUDA_MANAGED, test_size, true);

    // cuda_managed to ..
    test_copy(CUDA_MANAGED, HOST, test_size, true);
    test_copy(CUDA_MANAGED, DEVICE, test_size, true);
    test_copy(CUDA_MANAGED, MANAGED, test_size, true);
    test_copy(CUDA_MANAGED, CUDA_MANAGED, test_size, true);
#endif

    magmadnn_finalize();
    return 0;
}
