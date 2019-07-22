/**
 * @file tensor_io.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-04-05
 *
 * @copyright Copyright (c) 2019
 */
#include "tensor/tensor_io.h"

namespace magmadnn {
namespace io {

magmadnn_error_t read_csv_to_tensor(Tensor& t, const std::string& file_name, char delim) {
    magmadnn_error_t err = (magmadnn_error_t) 0;

    std::ifstream file_stream(file_name);

    if (!file_stream.is_open()) {
        /* error on opening file */
        return (magmadnn_error_t) 1;
    }

    std::stringstream ss;
    std::string line, token;
    int cur_idx = 0;

    unsigned int axes = t.shape().size();
    std::vector<index_t> indices(axes, 0);

    /* read in every line of the file */
    while (std::getline(file_stream, line)) {
        /* used to process the current line */
        ss.str(line);

        FOR_ALL_DTYPES(t.dtype(), Dtype, {
            Dtype val;

            /* read in each item separated by delim into num */
            while (std::getline(ss, token, delim)) {
                /* use iss's stream operator to correctly read value into val */
                if (!(std::istringstream(token) >> val)) {
                    /* could not read into val */
                    return (magmadnn_error_t) 2;
                }

                /* use tensor's built in flattened index set method */
                t.set<Dtype>(cur_idx, val);

                cur_idx++;
            }
        });

        ss.clear();
    }
    file_stream.close();

    return err;
}

magmadnn_error_t write_tensor_to_csv(const Tensor& t, const std::string& file_name, char delim, bool create) {
    magmadnn_error_t err = (magmadnn_error_t) 0;

    std::ofstream file_stream(file_name);
    if (!file_stream.is_open()) {
        /* failed to open file */
        return (magmadnn_error_t) 1;
    }

    FOR_ALL_DTYPES(t.dtype(), Dtype, {
        Dtype val;
        unsigned int size = t.get_size();

        for (unsigned int i = 0; i < size; i++) {
            val = t.get<Dtype>(i);
            if (!(file_stream << val << delim)) {
                /* for some reason errored while writing value */
                return (magmadnn_error_t) 2;
            }
        }
    });

    file_stream.flush();
    file_stream.close();

    return err;
}

}  // namespace io
}  // namespace magmadnn