#include "math/abs.h"


namespace magmadnn{
namespace math{

template <typename T>
void abs(Tensor<T>* x, Tensor<T>* out){
    if (out->get_memory_type() == HOST){
        assert(x->get_size() == out->get_size());
        for (unsigned i = 0; i < x->get_size();++i){
            out->set(i, std::abs(x->get(i)));
        }
    }
#if defined(_HAS_CUDA_)
    else {
        abs_device(x, out);
    }
#endif
}

template void abs(Tensor<int>* x, Tensor<int>* out);
template void abs(Tensor<float>* x, Tensor<float>* out);
template void abs(Tensor<double>* x, Tensor<double>* out);

}  //  namespace math
} //  namespace magmadnn