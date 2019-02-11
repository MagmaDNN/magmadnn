/*
    This is a test for the memorymanager class in skepsi
*/

#include <stdio.h>
#include <assert.h>
#include "skepsi.h"

using namespace skepsi;

void test_host_copy(unsigned int size) {
	
	printf("\nTesting host->host copy...  ");

	memorymanager<float> *m1 = new memorymanager<float> (size, HOST, (device_t) 0);
	memorymanager<float> *m2 = new memorymanager<float> (size, HOST, (device_t) 0);

	for (int i = 0; i < (int)m1->get_size(); i++) m1->set(i, 2*i);

	m2->copy_from(*m1);

	for (int i = 0; i < (int) m2->get_size(); i++) 
		assert( m1->get(i) == m2->get(i) );

	printf("Success!\n");

	printf("\nTesting host->device copy...  ");

	memorymanager<float> *m3 = new memorymanager<float> (size, DEVICE, (device_t) 0);

	m3->copy_from(*m1);

	printf("Success!\n");

	delete m1;
	delete m2;
	delete m3;
}


int main(int argc, char** argv) {

	unsigned int test_size = 10;

	if (argc == 2) test_size = atoi(argv[1]);

	test_host_copy(test_size);



    return 0;
}
