# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

##################################################################
# CUDA Toolkit libraries (including NVJPEG)
##################################################################
# Note: CUDA 8 support is unofficial.  CUDA 9 is officially supported
find_package(CUDA 8.0 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${CUDA_LIBRARIES})

list(APPEND DALI_EXCLUDES libcudart_static.a)

# For NVJPEG
if (BUILD_NVJPEG)
  find_package(NVJPEG 9.0 REQUIRED)
  if(${CUDA_VERSION} VERSION_LESS ${NVJPEG_VERSION})
    message(WARNING "Using nvJPEG ${NVJPEG_VERSION} together with CUDA ${CUDA_VERSION} "
                    "requires NVIDIA drivers compatible with CUDA ${NVJPEG_VERSION} or later")
  endif()
  include_directories(SYSTEM ${NVJPEG_INCLUDE_DIR})
  list(APPEND DALI_LIBS ${NVJPEG_LIBRARY})
  list(APPEND DALI_EXCLUDES libnvjpeg_static.a)
  add_definitions(-DDALI_USE_NVJPEG)

  if (${NVJPEG_LIBRARY_0_2_0})
    add_definitions(-DNVJPEG_LIBRARY_0_2_0)
  endif()

  if (${NVJPEG_DECOUPLED_API})
    add_definitions(-DNVJPEG_DECOUPLED_API)
  endif()
else()
  # Note: Support for disabling nvJPEG is unofficial
  message(STATUS "Building WITHOUT nvJPEG")
endif()

# NVIDIA NPPC library
find_cuda_helper_libs(nppc_static)

# NVIDIA NPPI library
if (${CUDA_VERSION} VERSION_LESS "9.0")
  # In CUDA 8, NPPI is a single library
  find_cuda_helper_libs(nppi_static)
  list(APPEND DALI_LIBS ${CUDA_nppi_static_LIBRARY})
  list(APPEND DALI_EXCLUDES libnppi_static.a)

else()

  find_cuda_helper_libs(nppicom_static)
  find_cuda_helper_libs(nppicc_static)
  find_cuda_helper_libs(nppig_static)
  list(APPEND DALI_LIBS ${CUDA_nppicom_static_LIBRARY}
    ${CUDA_nppicc_static_LIBRARY}
    ${CUDA_nppig_static_LIBRARY})
  list(APPEND DALI_EXCLUDES libnppicom_static.a
    libnppicc_static.a
    libnppig_static.a)
endif()
list(APPEND DALI_LIBS ${CUDA_nppc_static_LIBRARY})
list(APPEND DALI_EXCLUDES libnppc_static.a)

# CULIBOS needed when using static CUDA libs
find_cuda_helper_libs(culibos)
list(APPEND DALI_LIBS ${CUDA_culibos_LIBRARY})
list(APPEND DALI_EXCLUDES libculibos.a)

# NVTX for profiling
if (BUILD_NVTX)
  find_cuda_helper_libs(nvToolsExt)
  list(APPEND DALI_LIBS ${CUDA_nvToolsExt_LIBRARY})
  add_definitions(-DDALI_USE_NVTX)
endif()


include(cmake/Dependencies.common.cmake)

##################################################################
# protobuf
##################################################################
# link statically
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
list(APPEND DALI_LIBS ${Protobuf_LIBRARY})
# hide things from the protobuf, all we export is only is API generated from our proto files
list(APPEND DALI_EXCLUDES libprotobuf.a)


##################################################################
# Exclude stdlib
##################################################################
list(APPEND DALI_EXCLUDES libsupc++.a;libstdc++.a;libstdc++_nonshared.a;)


##################################################################
# Turing Optical flow API
##################################################################
if(BUILD_NVOF)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/turing_of)
endif()
