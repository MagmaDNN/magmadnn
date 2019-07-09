/**
 * @file tensor_io.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-04-05
 * 
 * @copyright Copyright (c) 2019
 */
#include <string>
#include <fstream>
#include <sstream>
#include "types.h"
#include "tensor.h"

namespace magmadnn {
namespace io {


    /** Reads from "file_name" into the tensor t. It interprets the data as having the flattened shape of the
     * input tensor. Returns 0 if successfull, otherwise something else. @see write_tensor_to_csv
     * @tparam T data type
     * @param t tensor to read values into
     * @param file_name file name of csv file (should be a text file, not binary)
     * @param delim the delimiter of the csv (assumed to be a comma)
     * @return magmadnn_error_t 0 if successful, otherwise anything else
     */
    template <typename T>
    magmadnn_error_t read_csv_to_tensor(Tensor<T>& t, const std::string& file_name, char delim=',');

    /** Writes the tensor t to the file "file_name". It writes a flattened version of the file that is readable
     * by read_csv_to_tensor. @see read_csv_to_tensor .
     * @tparam T data type
     * @param t tensor to write out
     * @param file_name csv file to be written into. Created if it does not exist.
     * @param delim character to delimit values
     * @param create create file if it does not exist.
     * @return magmadnn_error_t returns 0 if successful, otherwise something else
     */
    template <typename T>
    magmadnn_error_t write_tensor_to_csv(const Tensor<T>& t, const std::string& file_name, char delim=',', bool create=true);


}   // namespace io
}   // namespace io