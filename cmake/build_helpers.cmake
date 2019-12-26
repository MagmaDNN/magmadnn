function(magmadnn_default_includes name)
  # set include path depending on used interface
  target_include_directories("${name}"
    PUBLIC
    $<BUILD_INTERFACE:${MagmaDNN_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${MagmaDNN_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${MagmaDNN_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )
  
  if (MAGMADNN_ENABLE_CUDA)
    target_include_directories("${name}" PRIVATE ${CUDA_INCLUDE_DIRS}) 
    target_include_directories("${name}" PRIVATE ${MAGMA_INCLUDE_DIRS}) 
  endif ()

endfunction()

function(magmadnn_compile_features name)
  target_compile_features("${name}" PUBLIC cxx_std_11)
  set_target_properties("${name}" PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

function(magmadnn_add_example tests_driver)

  get_filename_component(tests_driver_name ${tests_driver} NAME_WE)
  add_executable(${tests_driver_name} ${tests_driver})
  if (MAGMADNN_ENABLE_CUDA)
    target_include_directories(${tests_driver_name} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_include_directories(${tests_driver_name} PRIVATE ${MAGMA_INCLUDE_DIRS}) 
  endif ()
  target_link_libraries(${tests_driver_name} PRIVATE magmadnn)
  target_link_libraries(${tests_driver_name} PRIVATE ${LIBS})
  
endfunction()
