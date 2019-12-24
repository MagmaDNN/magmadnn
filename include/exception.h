#pragma once

namespace magmadnn {

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
