/**
 * @file tensor_io.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-04-05
 * 
 * @copyright Copyright (c) 2019
 */
#include "tensor/tensor_io.h"

namespace skepsi {
namespace io {


    template <typename T>
    skepsi_error_t read_csv_to_tensor(Tensor<T>& t, const std::string& file_name, char delim) {
        skepsi_error_t err = (skepsi_error_t) 0;

        std::ifstream file_stream (file_name);

        if (!file_stream.is_open()) {
            /* error on opening file */
            return (skepsi_error_t) 1;
        }

        std::stringstream ss;
        std::string line, token;
        T val;
        int cur_idx = 0;

        unsigned int axes = t.get_shape().size();
        std::vector<unsigned int> indices (axes, 0);

        /* read in every line of the file */
        while (std::getline(file_stream, line)) {
            /* used to process the current line */
            ss.str(line);

            /* read in each item separated by delim into num */
            while (std::getline(ss, token, delim)) {
                /* use iss's stream operator to correctly read value into val */
                if (!(std::istringstream (token) >> val)) {
                    /* could not read into val */
                    return (skepsi_error_t) 2;
                }

                /* use tensor's built in flattened index set method */
                t.set(cur_idx, val);

                cur_idx++;
            }

            ss.clear();
        }
        file_stream.close();

        return err;
    }
    template skepsi_error_t read_csv_to_tensor(Tensor<int>&, const std::string&, char);
    template skepsi_error_t read_csv_to_tensor(Tensor<float>&, const std::string&, char);
    template skepsi_error_t read_csv_to_tensor(Tensor<double>&, const std::string&, char);

    template <typename T>
    skepsi_error_t write_tensor_to_csv(const Tensor<T>& t, const std::string& file_name, char delim, bool create) {
        skepsi_error_t err = (skepsi_error_t) 0;

        std::ofstream file_stream (file_name);
        if (!file_stream.is_open()) {
            /* failed to open file */
            return (skepsi_error_t) 1;
        }

        T val;
        unsigned int size = t.get_size();
        
        for (unsigned int i = 0; i < size; i++) {
            val = t.get(i);
            if (!(file_stream << val << delim)) {
                /* for some reason errored while writing value */
                return (skepsi_error_t) 2;
            }
        }

        file_stream.flush();
        file_stream.close();

        return err;
    }
    template skepsi_error_t write_tensor_to_csv(const Tensor<int>&, const std::string&, char, bool);
    template skepsi_error_t write_tensor_to_csv(const Tensor<float>&, const std::string&, char, bool);
    template skepsi_error_t write_tensor_to_csv(const Tensor<double>&, const std::string&, char, bool);


}   // namespace io
}   // namespace io