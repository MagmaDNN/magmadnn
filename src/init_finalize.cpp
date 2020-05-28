/**
 * @file init_finalize.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-03-15
 *
 * @copyright Copyright (c) 2019
 */
#include "magmadnn/init_finalize.h"

#if defined(MAGMADNN_HAVE_CUDA)
#include <cuda.h>
#endif

namespace magmadnn {

namespace internal {
magmadnn_settings_t *MAGMADNN_SETTINGS;
}  // namespace internal

magmadnn_error_t magmadnn_init() {
    magmadnn_error_t err = 0;

    /* init the settings struct */
    internal::MAGMADNN_SETTINGS = new magmadnn_settings_t;

#if defined(MAGMADNN_HAVE_CUDA)
    err = (magmadnn_error_t) magma_init();
    
    // TODO: Create stream and init handles with it. DO NOT use
    // default stream
    /* init cudnn */
    cudnnCreate(&internal::MAGMADNN_SETTINGS->cudnn_handle);

    /* init cublas */
    cublasCreate(&internal::MAGMADNN_SETTINGS->cublas_handle);

    internal::MAGMADNN_SETTINGS->n_devices = 1; /* TODO : read in number of devices */

    int rank = 0;
#if defined(MAGMADNN_HAVE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    int num_devices;

    // query number of devices                                                                  
    cudaError_t cuerr;
    cuerr = cudaGetDeviceCount( &num_devices );
    assert( cuerr == 0 || cuerr == cudaErrorNoDevice );
    cudaSetDevice(rank%num_devices); 
#endif

    return err;
}

magmadnn_error_t magmadnn_finalize() {
    magmadnn_error_t err = 0;

#if defined(MAGMADNN_HAVE_CUDA)
    err = (magmadnn_error_t) magma_finalize();

    /* destroy cudnn */
    cudnnDestroy(internal::MAGMADNN_SETTINGS->cudnn_handle);

    /* destroy cublas */
    cublasDestroy(internal::MAGMADNN_SETTINGS->cublas_handle);
#endif

    /* delete settings */
    delete internal::MAGMADNN_SETTINGS;

    return err;
}

}  // namespace magmadnn
