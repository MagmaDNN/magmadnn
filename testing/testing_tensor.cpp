#include "skepsi.h"

using namespace skepsi;

void test_indexing();

int main(int argc, char **argv) {
    
    test_indexing();

    return 0;
}

void test_indexing() {
    tensor<float> *t = new tensor<float> ({4, 3, 2});

    // test
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 2; k++)
                t->set({i,j,k}, i*j*k);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 2; k++)
                printf("t[%d, %d, %d] = %.2f\n", i, j, k, t->get({i,j,k}));
    
    delete t;
}
