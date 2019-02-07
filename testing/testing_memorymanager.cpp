/*
    This is a test for the memorymanager class in danielnn
*/

#include <stdio.h>
#include "danielnn.h"

int main(int argc, char** argv) {

    memorymanager<float> *mm = new memorymanager<float> (10, HOST, 0);

    // set values of memory manager
    for (int i = 0; i < mm->get_size(); i++) {
        mm->set(i, i*i * (0.333));
    }

    // print the values
    for (int i = 0; i < mm->get_size(); i++) {
        printf("%d: %.3f\n", i, mm->get(i));
    }

    delete mm;
    return 0;
}