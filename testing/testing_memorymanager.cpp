/*
    This is a test for the memorymanager class in danielnn
*/

#include <stdio.h>
#include "danielnn.h"

int main(int argc, char** argv) {

    int size = 4;

    memorymanager<float> *mm = new memorymanager<float> (size, HOST, 0);
    float *arr = new float[size];

    // set values of memory manager
    for (int i = 0; i < mm->get_size(); i++) {
        mm->set(i, i*i * (0.333));
        arr[i] = 0.5 * i;
    }

    // print the values
    for (int i = 0; i < mm->get_size(); i++) {
        printf("%d: %.3f\n", i, mm->get(i));
    }

    // copy in new vals
    mm->copy_from_host(arr);
    printf("\nnew vals:\n");
    // print the values
    for (int i = 0; i < mm->get_size(); i++) {
        printf("%d: %.3f\n", i, mm->get(i));
    }

    delete mm;
    return 0;
}