cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(kokoro)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2")
endif()

include(cmake/msp_dependencies.cmake)
add_definitions(-DENV_HAS_STD_FILESYSTEM)
add_definitions(-DENV_HAS_POSIX_FILE_STAT)

include_directories(${MSP_INC_DIR})
link_directories(${MSP_LIB_DIR})

# onnxruntime
include_directories(third_party/onnxruntime/include)
include_directories(third_party/onnxruntime/include/onnxruntime/core/session)
link_directories(third_party/onnxruntime/lib)
list(APPEND ORT_LIBS onnxruntime onnxruntime_providers_shared)

# limonp
include_directories(src/limonp/include)

# Eigen
include_directories(src/librosa/eigen3)

# librosa
include_directories(src/librosa/)

# espeak-ng
include_directories(third_party/espeak-ng/include)
link_directories(third_party/espeak-ng/lib)
list(APPEND ESPEAK_LIBS libespeak-ng.a pthread)

include_directories(src)
aux_source_directory(src SRC)
aux_source_directory(src/ax_model_runner SRC)

add_executable(kokoro main.cpp ${SRC})
target_link_libraries(kokoro ${MSP_LIBS} ${ORT_LIBS} ${ESPEAK_LIBS})

add_executable(test_ax_model tests/test_ax_model.cpp ${SRC})
target_link_libraries(test_ax_model ${MSP_LIBS} ${ORT_LIBS} ${ESPEAK_LIBS})

add_executable(test_jieba tests/test_jieba.cpp ${SRC})
target_link_libraries(test_jieba ${MSP_LIBS} ${ORT_LIBS} ${ESPEAK_LIBS})

add_executable(test_espeak tests/test_espeak.cpp ${SRC})
target_link_libraries(test_espeak ${MSP_LIBS} ${ORT_LIBS} ${ESPEAK_LIBS})

add_executable(kokoro_srv kokoro_srv.cpp ${SRC})
target_link_libraries(kokoro_srv ${MSP_LIBS} ${ORT_LIBS} ${ESPEAK_LIBS})

install(TARGETS 
        kokoro 
        test_ax_model
        test_jieba
        kokoro_srv
        test_espeak
    RUNTIME
        DESTINATION .)

set_target_properties(kokoro
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/../third_party/onnxruntime/lib"
)
set_target_properties(test_ax_model
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/../third_party/onnxruntime/lib"
)  
set_target_properties(test_jieba
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/../third_party/onnxruntime/lib"
)  
set_target_properties(kokoro_srv
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/../third_party/onnxruntime/lib"
)  
set_target_properties(test_espeak
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/../third_party/onnxruntime/lib"
)
