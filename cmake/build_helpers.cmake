function(magmadnn_default_includes name)
  # set include path depending on used interface
  target_include_directories("${name}"
    PUBLIC
    $<BUILD_INTERFACE:${MagmaDNN_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${MagmaDNN_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${MagmaDNN_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )
endfunction()

function(magmadnn_compile_features name)
  target_compile_features("${name}" PUBLIC cxx_std_11)
  set_target_properties("${name}" PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()
