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

include(GNUInstallDirs)
set(DNNL_LIBDIR "lib")
if(CMAKE_INSTALL_LIBDIR MATCHES ".*lib64$")
  set(DNNL_LIBDIR "lib64")
endif()

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

  if (MAGMADNN_ENABLE_MKLDNN)
    install(DIRECTORY "${MagmaDNN_BINARY_DIR}/third_party/mkldnn/build/include/"
      DESTINATION "${MAGMADNN_INSTALL_INCLUDE_DIR}/third_party/mkldnn"
      FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY "${MagmaDNN_BINARY_DIR}/third_party/mkldnn/build/include/"
      DESTINATION "${MAGMADNN_INSTALL_INCLUDE_DIR}/third_party/mkldnn"
      FILES_MATCHING PATTERN "*.hpp")
    install(
      FILES "${MagmaDNN_BINARY_DIR}/third_party/mkldnn/build/${DNNL_LIBDIR}/libdnnl.so.1.2"
      "${MagmaDNN_BINARY_DIR}/third_party/mkldnn/build/${DNNL_LIBDIR}/libdnnl.so.1"
      "${MagmaDNN_BINARY_DIR}/third_party/mkldnn/build/${DNNL_LIBDIR}/libdnnl.so"
      DESTINATION "${MAGMADNN_INSTALL_LIBRARY_DIR}/third_party/mkldnn"
      )
    #    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \
    # ${MagmaDNN_BINARY_DIR}/third_party/mkldnn/build/lib/libdnnl.so.1.2 \
    # ${MagmaDNN_BINARY_DIR}/third_party/mkldnn/build/lib/libdnnl.so)")

  endif()
  
endfunction()
