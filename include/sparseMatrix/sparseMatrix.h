#pragma once

#include <vector>
#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cusparse.h"
#define CUSPARSE_USE_NEW_API_VERSION 100100

#endif

namespace magmadnn {
namespace spMatrix {

typedef enum storage_format_t { DENSE, CSR } storage_format_t;

typedef enum lib_format_t {
    NATIVE,
#if defined(_HAS_CUDA_)
    CUSPARSE,
//  MAGMA  //  not yet supported
#endif
} lib_format_t;

template <typename T>
class sparseMatrix {
   private:
    memory_t _mem_type;
    storage_format_t _storage_format;
    lib_format_t _lib_format;
    unsigned _dim0, _dim1;
    unsigned _nnz;
#if defined(_HAS_CUDA_)
    cudaDataType _data_type;
    void set_cuda_data_type(void)
#endif
    union {
        struct {
            Tensor<T> *matrix;
        } dense;
        struct {
            Tensor<T> *val;
            Tensor<T> *rowAcc;
            Tensor<T> *colIdx;
        } csr;
    } data;
    union {
        struct {
#if defined(_HAS_CUDA_)
#if (CUDART_VERSION >= CUSPARSE_USE_NEW_API_VERSION)
			//  descriptor for new API
            cusparseDnMatDescr_t cuSpDnMat_desc;
#else
            //  descriptor for old API
            /* empty */
#endif
#endif
        } dense;
        struct {
#if defined(_HAS_CUDA_)
#if (CUDART_VERSION >= CUSPARSE_USE_NEW_API_VERSION)
            cusparseSpMatDescr_t cuSpSpMat_desc;
#else
            cusparseMatDescr_t cuSpMat_desc;
#endif
#endif
        } sparse;
    } descriptor;

   public:
    sparseMatrix(void);
    sparseMatrix(const Tensor<T> &matrixTensor, memory_t mem_type, bool copy, storage_format_t storage_format = DENSE,
                 lib_format_t lib_format = NATIVE);
    sparseMatrix(unsigned dim0, unsigned dim1, memory_t mem_type, storage_format_t storage_format = DENSE,
                 lib_format_t lib_format = NATIVE);
    ~sparseMatrix(void);
    inline storage_format_t get_storage_format(void) const { return _storage_format; }
    inline lib_format_t get_lib_format(void) const { return _lib_format; }
    inline memory_t get_mem_type(void) const { return _mem_type; }
    inline unsigned get_shape(unsigned idx) {
        assert(idx == 0 || idx == 1);
        return idx == 0 ? _dim0 : _dim1;
    }
    inline std::vector<unsigned> get_shape(void) const { return std::vector<unsigned>{_dim0, _dim1}; }
    inline unsigned get_nnz(void) const { return _nnz; }

};  // namespace spMatrix

}  // namespace spMatrix
}  // namespace magmadnn

#undef CUSPARSE_USE_NEW_API_VERSION
