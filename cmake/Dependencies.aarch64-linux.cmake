#############################
#        CUDA TOOLKIT
#############################

message(STATUS "CMAKE_CROSSCOMPILING ${CMAKE_CROSSCOMPILING}")


set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_HOST})
set(CUDA_TOOLKIT_TARGET_DIR ${CUDA_TARGET})

find_package(CUDA 10.0 REQUIRED)
message(STATUS "CUDA_VARIABLES:")

# message(STATUS "CUDA_VERSION_MAJOR: ${CUDA_VERSION_MAJOR}")
# message(STATUS "CUDA_VERSION_MINOR: ${CUDA_VERSION_MINOR}")
# message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
# message(STATUS "CUDA_VERSION_STRING: ${CUDA_VERSION_STRING}")
# message(STATUS "CUDA_HAS_FP16: ${CUDA_HAS_FP16}")
# message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
# message(STATUS "CUDA_SDK_ROOT_DIR: ${CUDA_SDK_ROOT_DIR}")
# message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
# message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
# message(STATUS "CUDA_CUFFT_LIBRARIES: ${CUDA_CUFFT_LIBRARIES}")
# message(STATUS "CUDA_CUBLAS_LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")
# message(STATUS "CUDA_cudart_static_LIBRARY: ${CUDA_cudart_static_LIBRARY}")
# message(STATUS "CUDA_cudadevrt_LIBRARY: ${CUDA_cudadevrt_LIBRARY}")
# message(STATUS "CUDA_cupti_LIBRARY: ${CUDA_cupti_LIBRARY}")
# message(STATUS "CUDA_curand_LIBRARY: ${CUDA_curand_LIBRARY}")
# message(STATUS "CUDA_cusolver_LIBRARY: ${CUDA_cusolver_LIBRARY}")
# message(STATUS "CUDA_cusparse_LIBRARY: ${CUDA_cusparse_LIBRARY}")
# message(STATUS "CUDA_npp_LIBRARY: ${CUDA_npp_LIBRARY}")
# message(STATUS "CUDA_nppc_LIBRARY: ${CUDA_nppc_LIBRARY}")
# message(STATUS "CUDA_nppi_LIBRARY: ${CUDA_nppi_LIBRARY}")
# message(STATUS "CUDA_nppial_LIBRARY: ${CUDA_nppial_LIBRARY}")
# message(STATUS "CUDA_nppicc_LIBRARY: ${CUDA_nppicc_LIBRARY}")
# message(STATUS "CUDA_nppicom_LIBRARY: ${CUDA_nppicom_LIBRARY}")
# message(STATUS "CUDA_nppidei_LIBRARY: ${CUDA_nppidei_LIBRARY}")
# message(STATUS "CUDA_nppif_LIBRARY: ${CUDA_nppif_LIBRARY}")
# message(STATUS "CUDA_nppig_LIBRARY: ${CUDA_nppig_LIBRARY}")
# message(STATUS "CUDA_nppim_LIBRARY: ${CUDA_nppim_LIBRARY}")
# message(STATUS "CUDA_nppist_LIBRARY: ${CUDA_nppist_LIBRARY}")
# message(STATUS "CUDA_nppisu_LIBRARY: ${CUDA_nppisu_LIBRARY}")
# message(STATUS "CUDA_nppitc_LIBRARY: ${CUDA_nppitc_LIBRARY}")
# message(STATUS "CUDA_npps_LIBRARY: ${CUDA_npps_LIBRARY}")
# message(STATUS "CUDA_nvcuvenc_LIBRARY: ${CUDA_nvcuvenc_LIBRARY}")
# message(STATUS "CUDA_nvcuvid_LIBRARY: ${CUDA_nvcuvid_LIBRARY}")

# message(STATUS "CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR}")
# message(STATUS "CUDA_TOOLKIT_TARGET_DIR ${CUDA_TOOLKIT_TARGET_DIR}")

include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${CUDA_LIBRARIES})

# NVIDIA NPPC library
find_cuda_helper_libs(nppc_static)
find_cuda_helper_libs(nppicom_static)
find_cuda_helper_libs(nppicc_static)
find_cuda_helper_libs(nppig_static)
list(APPEND DALI_LIBS ${CUDA_nppicom_static_LIBRARY}
  ${CUDA_nppicc_static_LIBRARY}
  ${CUDA_nppig_static_LIBRARY})
list(APPEND DALI_EXCLUDES libnppicom_static.a
  libnppicc_static.a
  libnppig_static.a)
list(APPEND DALI_LIBS ${CUDA_nppc_static_LIBRARY})
list(APPEND DALI_EXCLUDES libnppc_static.a)

# CULIBOS needed when using static CUDA libs
find_cuda_helper_libs(culibos)
list(APPEND DALI_LIBS ${CUDA_culibos_LIBRARY})
list(APPEND DALI_EXCLUDES libculibos.a)


message(STATUS "CUDA_nppc_static_LIBRARY: ${CUDA_nppc_static_LIBRARY}")
message(STATUS "CUDA_nppicom_static_LIBRARY: ${CUDA_nppicom_static_LIBRARY}")
message(STATUS "CUDA_nppicc_static_LIBRARY: ${CUDA_nppicc_static_LIBRARY}")
message(STATUS "CUDA_nppig_static_LIBRARY: ${CUDA_nppig_static_LIBRARY}")
message(STATUS "CUDA_culibos_LIBRARY: ${CUDA_culibos_LIBRARY}")

include_directories(${CUDA_TOOLKIT_TARGET_DIR}/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CUDA_TARGET_LIBRARIES_DIR} -L${CUDA_TARGET_LIBRARIES_DIR}/stubs -lcudart -lnppc_static -lnppicom_static -lnppicc_static -lnppig_static -lculibos")

# NVTX for profiling
if (BUILD_NVTX)
  list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/libnvToolsExt.so)
  add_definitions(-DDALI_USE_NVTX)
endif()

##################################################################
# Common dependencies
##################################################################
include(cmake/Dependencies.common.cmake)

##################################################################
# protobuf
##################################################################
set(Protobuf_CROSS YES)
set(Protobuf_USE_STATIC_LIBS YES)
find_package(Protobuf 2.0 REQUIRED)
if(${Protobuf_VERSION} VERSION_LESS "3.0")
  message(STATUS "TensorFlow TFRecord file format support is not available with Protobuf 2")
else()
  message(STATUS "Enabling TensorFlow TFRecord file format support")
  add_definitions(-DDALI_BUILD_PROTO3=1)
  set(BUILD_PROTO3 ON CACHE STRING "Build proto3")
endif()

include_directories(SYSTEM ${Protobuf_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${Protobuf_LIBRARY} ${Protobuf_PROTOC_LIBRARIES} ${Protobuf_LITE_LIBRARIES})
