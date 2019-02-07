/*
    This is a test for the memorymanager class in danielnn
*/

#include <stdio.h>
#include "danielnn.h"

int main(int argc, char** argv) {

    memorymanager<float> *mm = new memorymanager<float> (10, HOST, 0);

    delete mm;
    return 0;
}