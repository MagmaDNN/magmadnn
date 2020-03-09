#pragma once

#include "magmadnn/types.h"

#include <exception>
#include <string>

namespace magmadnn {

class Error : public std::exception {
public:
    /**
     * Initializes an error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param what  The error message
     */
    Error(const std::string &file, int line, const std::string &what)
        : what_(file + ":" + std::to_string(line) + ": " + what)
    {}

    /**
     * Returns a human-readable string with a more detailed description of the
     * error.
     */
    virtual const char *what() const noexcept override { return what_.c_str(); }

private:
    const std::string what_;
};

   
/**
 * NotImplemented is thrown in case an operation has not yet
 * been implemented (but will be implemented in the future).
 */
class NotImplemented : public Error {
public:
    /**
     * Initializes a NotImplemented error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the not-yet implemented function
     */
    NotImplemented(const std::string &file, int line, const std::string &func)
        : Error(file, line, func + " is not implemented")
    {}
};

/**
 * CudaError is thrown when a CUDA routine throws a non-zero error code.
 */
class CudaError : public Error {
public:
    /**
     * Initializes a CUDA error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the CUDA routine that failed
     * @param error_code  The resulting CUDA error code
     */
    CudaError(const std::string &file, int line, const std::string &func,
              int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};
   
/**
 * CublasError is thrown when a cuBLAS routine throws a non-zero error code.
 */
class CublasError : public Error {
public:
    /**
     * Initializes a cuBLAS error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the cuBLAS routine that failed
     * @param error_code  The resulting cuBLAS error code
     */
    CublasError(const std::string &file, int line, const std::string &func,
                int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};

/**
 * CudnnError is thrown when a cuDNN routine throws a non-zero error code.
 */
class CudnnError : public Error {
public:
    /**
     * Initializes a cuBLAS error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the cuBLAS routine that failed
     * @param error_code  The resulting cuBLAS error code
     */
    CudnnError(const std::string &file, int line, const std::string &func,
                int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};

} // magmadnn namespace
