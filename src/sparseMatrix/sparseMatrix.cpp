#include "sparseMatrix/sparseMatrix.h"

#if defined(_HAS_CUDA_)
#define CUSPARSE_USE_NEW_API_VERSION 100100
#endif

namespace magmadnn {
namespace spMatrix {

#if defined(_HAS_CUDA_)
template <typename T>
void sparseMatrix<T>::set_cuda_data_type(void) {
    std::fprintf(stderr, "Data type not supported.\n");
}
template <>
void sparseMatrix<float>::set_cuda_data_type(void) {
    this->_data_type = CUDA_R_32F;
}
template <>
void sparseMatrix<double>::set_cuda_data_type(void) {
    this->_data_type = CUDA_R_64F;
}
#endif

template <typename T>
sparseMatrix<T>::sparseMatrix(void)
    : _mem_type(HOST), _storage_format(DENSE), _lib_format(NATIVE), _dim0(0), _dim1(0), _nnz(0) {
    this->data.dense.matrix = nullptr;
}

template <typename T>
sparseMatrix<T>::sparseMatrix(const Tensor<T> &matrixTensor, memory_t mem_type, bool copy,
                              storage_format_t storage_format, lib_format_t lib_format)
    : _mem_type(mem_type), _storage_format(storage_format), _lib_format(lib_format) {
    assert(T_IS_MATRIX(&matrixTensor));
    _dim0 = matrixTensor.get_shape(0);
    _dim1 = matrixTensor.get_shape(1);
    _nnz = 0;
    if (storage_format == DENSE) {
        this->data.dense.matrix = new Tensor<T>({_dim0, _dim1}, {ZERO, {}}, mem_type);
        if (copy) {
            this->data.dense.matrix.copy_from(matrixTensor);
            for (unsigned i = 0; i < _dim0; i++) {
                for (unsigned j = 0; j < _dim1; j++) {
                    if (this->data.dense.matrix->get({i, j}) != (T) 0) {
                        ++this->_nnz;
                    }
                }
            }
        }
#if defined(_HAS_CUDA_)
        if (lib_format == CUSPARSE) {
            this->set_cuda_data_type();
#if (CUDART_VERSION >= CUSPARSE_USE_NEW_API_VERSION)
            //  descriptor for new API
            assert(_mem_type != HOST);
            cusparseErrchk(cusparseCreateDnMat(&this->descriptor.dense.cuSpDnMat_desc, _dim1, _dim0, _dim1,
                                               this->data.dense.matrix->get_ptr(), _data_type, CUSPARSE_ORDER_ROW));
#else
            //  descriptor for old API
            /* empty */
#endif
        }
#endif
    } else if (storage_format == CSR) {
        std::vector<T> valV;
        std::vector<int> rowAccV, colIdxV;
        rowAccV.push_back(0);
        unsigned rowCounter = 0;
        T v;
        for (unsigned i = 0; i < _dim0; i++) {
            for (unsigned j = 0; j < _dim1; ++j) {
                v = matrixTensor.get({i, j});
                if (v != (T) 0) {
                    valV.push_back(v);
                    colIdxV.push_back(j);
                    ++rowCounter;
                }
            }
            rowAccV.push_back(rowCounter);
        }
        _nnz = rowCounter;
        this->data.csr.val = new Tensor<T>({_nnz}, {NONE, {}}, mem_type);
        this->data.csr.rowAcc = new Tensor<int>({_dim0 + 1}, {NONE, {}}, mem_type);
        this->data.csr.colIdx = new Tensor<int>({_nnz}, {NONE, {}}, mem_type);
        //  todo: add batch assign
        for (unsigned idx = 0; idx < _nnz; ++idx) {
            this->data.csr.val->set(idx, valV[idx]);
            this->data.csr.colIdx->set(idx, colIdxV[idx]);
        }
        for (unsigned idx = 0; idx < _dim0; ++idx) {
            this->data.csr.rowAcc->set(idx, rowAccV[idx]);
        }
#if defined(_HAS_CUDA_)
        if (lib_format == CUSPARSE) {
            assert(_mem_type != HOST);
            this->set_cuda_data_type();
#if (CUDART_VERSION >= CUSPARSE_USE_NEW_API_VERSION)
            //  descriptor for new API
            cusparseErrchk(cusparseCreateCsr(&this->descriptor.sparse.cuSpMat_desc, _dim0, _dim1, _nnz,
                                             this->_data.csr.rowAcc->get_ptr(), this->_data.csr.colIdx->get_ptr(),
                                             this->_data.csr.val->get_ptr(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO, _data_type));
#else
            //  descriptor for old API
            cusparseErrchk(cusparseCreateMatDescr(&this->descriptor.sparse.cuSpMat_desc));
            cusparseErrchk(cusparseSetMatIndexBase(this->descriptor.sparse.cuSpMat_desc, CUSPARSE_INDEX_BASE_ZERO));
            cusparseErrchk(cusparseSetMatType(this->descriptor.sparse.cuSpMat_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
#endif
        }
#endif
    } else {
        std::fprintf(stderr, "Requested storage format for sparse matrix is not recongnized.\n");
    }
}

template <typename T>
sparseMatrix<T>::~sparseMatrix(void) {
    switch (this->_storage_format) {
        case DENSE:
            delete this->data.dense.matrix;
#if defined(_HAS_CUDA_)
            if (_lib_format == CUSPARSE) {
#if (CUDART_VERSION >= CUSPARSE_USE_NEW_API_VERSION)
                cusparseErrchk(cusparseDestroyDnMat(this->descriptor.dense.cuSpDnMat_desc));
#endif
			}
#endif
            break;
        case CSR:
            delete this->data.csr.val;
            delete this->data.csr.rowAcc;
            delete this->data.csr.colIdx;
            break;
        default:
            std::fprintf(stderr, "Unknown type for storing sparse matrix.\n");
            break;
    }
}

}  //  namespace spMatrix
}  //  namespace magmadnn

#undef CUSPARSE_USE_NEW_API_VERSION