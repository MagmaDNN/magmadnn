function(magmadnn_add_test tests_driver)

  get_filename_component(tests_driver_name ${tests_driver} NAME_WE)
  add_executable(${tests_driver_name} ${tests_driver})
  if (MAGMADNN_ENABLE_CUDA)
    target_include_directories(${tests_driver_name} PRIVATE ${CUDNN_INCLUDE_DIRS}) 
    target_include_directories(${tests_driver_name} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_include_directories(${tests_driver_name} PRIVATE ${MAGMA_INCLUDE_DIRS}) 
  endif ()
  if (MAGMADNN_ENABLE_MKLDNN)
    target_include_directories(${tests_driver_name} PRIVATE ${MKLDNN_INCLUDE_DIRS}) 
  endif ()
  target_link_libraries(${tests_driver_name} PRIVATE magmadnn)
  target_link_libraries(${tests_driver_name} PRIVATE ${LIBS})

  add_test(NAME ${tests_driver_name} COMMAND ${tests_driver_name})

endfunction()
