set(MAGMADNN_INSTALL_INCLUDE_DIR "include")
set(MAGMADNN_INSTALL_LIBRARY_DIR "lib")
set(MAGMADNN_INSTALL_PKGCONFIG_DIR "lib/pkgconfig")
set(MAGMADNN_INSTALL_CONFIG_DIR "lib/cmake/MagmaDNN")
set(MAGMADNN_INSTALL_MODULE_DIR "lib/cmake/MagmaDNN/Modules")

# function(magmadnn_install_library name subdir)
function(magmadnn_install_library name)
    # install .so and .a files
    install(TARGETS "${name}"
        EXPORT MagmaDNN
        LIBRARY DESTINATION ${MAGMADNN_INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${MAGMADNN_INSTALL_LIBRARY_DIR}
        )
endfunction()

function(magmadnn_install)

  # install the public header files
  install(DIRECTORY "${MagmaDNN_SOURCE_DIR}/include/"
    DESTINATION "${MAGMADNN_INSTALL_INCLUDE_DIR}"
    FILES_MATCHING PATTERN "*.h"
    )
  install(DIRECTORY "${MagmaDNN_BINARY_DIR}/include/"
    DESTINATION "${MAGMADNN_INSTALL_INCLUDE_DIR}"
    FILES_MATCHING PATTERN "*.h"
    )

endfunction()
