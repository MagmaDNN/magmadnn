#pragma once

#include <exception>

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

} // magmadnn namespace
