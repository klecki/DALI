#############################
#        CUDA TOOLKIT
#############################

find_package(CUDA 10.0 REQUIRED)

set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_HOST})
set(CUDA_TOOLKIT_TARGET_DIR ${CUDA_TARGET})
set(CUDA_TOOLKIT_ROOT_DIR_INTERNAL ${CUDA_TOOLKIT_ROOT_DIR})
set(CUDA_TOOLKIT_TARGET_DIR_INTERNAL ${CUDA_TOOLKIT_TARGET_DIR})
set(CUDA_TARGET_LIBRARIES_DIR ${CUDA_TOOLKIT_TARGET_DIR}/lib)
set(CUDA_LIBRARIES "") # WIPE THE CUDA_LIBRARIES used by CMAKE, and pass the flags by hand

list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/libcudart.so)
list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/libnppc_static.a)
list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/libnppicom_static.a)
list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/libnppicc_static.a)
list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/libnppig_static.a)
list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/libculibos.a)
list(APPEND DALI_LIBS ${CMAKE_DL_LIBS})

list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/stubs/libnppc.so)
list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/stubs/libnppicom.so)
list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/stubs/libnppicc.so)
list(APPEND DALI_LIBS ${CUDA_TARGET_LIBRARIES_DIR}/stubs/libnppig.so)

include_directories(${CUDA_TOOLKIT_TARGET_DIR}/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CUDA_TARGET_LIBRARIES_DIR} -L${CUDA_TARGET_LIBRARIES_DIR}/stubs -lcudart -lnppc_static -lnppicom_static -lnppicc_static -lnppig_static -lnpps -lnppc -lculibos")

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
message(STATUS "PROTOBUF CALL 2")
if(${Protobuf_VERSION} VERSION_LESS "3.0")
  message(STATUS "TensorFlow TFRecord file format support is not available with Protobuf 2")
else()
  message(STATUS "Enabling TensorFlow TFRecord file format support")
  add_definitions(-DDALI_BUILD_PROTO3=1)
  set(BUILD_PROTO3 ON CACHE STRING "Build proto3")
endif()

include_directories(SYSTEM ${Protobuf_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${Protobuf_LIBRARY} ${Protobuf_PROTOC_LIBRARIES} ${Protobuf_LITE_LIBRARIES})
