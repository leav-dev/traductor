cmake_minimum_required(VERSION 3.16)
project(sussurro LANGUAGES C CXX)

# SentencePiece (fetched below) ships an old cmake_minimum_required; allow it under CMake 4+
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml)

include(FetchContent)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG v0.2.0
)
FetchContent_MakeAvailable(sentencepiece)

# whisper.cpp reuses the ggml target added above (its CMake guards with `if (NOT TARGET ggml)`)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/whisper.cpp)

add_library(sussurro_core STATIC
    src/model.cpp
    src/tokenizer.cpp
    src/encoder.cpp
    src/decoder.cpp)
target_include_directories(sussurro_core PUBLIC src)
target_include_directories(sussurro_core PRIVATE ${sentencepiece_SOURCE_DIR}/src)
target_link_libraries(sussurro_core PUBLIC ggml PRIVATE sentencepiece-static)

add_executable(sussurro src/main.cpp)
target_link_libraries(sussurro PRIVATE sussurro_core)

add_executable(sussurro-quantize src/quantize.cpp)
target_link_libraries(sussurro-quantize PRIVATE ggml)

# speech-to-text -> translation (whisper.cpp + sussurro_core)
add_executable(sussurro-interpret src/interpret.cpp)
target_link_libraries(sussurro-interpret PRIVATE sussurro_core whisper)

# text-to-speech via sherpa-onnx (prebuilt C API, downloaded by the user; see README)
set(SHERPA_DIR ${CMAKE_SOURCE_DIR}/third_party/sherpa-onnx)
if(EXISTS ${SHERPA_DIR}/include/sherpa-onnx/c-api/c-api.h)
    add_executable(sussurro-speak src/speak.cpp)
    target_include_directories(sussurro-speak PRIVATE ${SHERPA_DIR}/include)
    target_link_directories(sussurro-speak PRIVATE ${SHERPA_DIR}/lib)
    target_link_libraries(sussurro-speak PRIVATE sherpa-onnx-c-api)
    set_target_properties(sussurro-speak PROPERTIES
        BUILD_RPATH ${SHERPA_DIR}/lib
        INSTALL_RPATH ${SHERPA_DIR}/lib)
else()
    message(STATUS "sherpa-onnx not found in third_party/sherpa-onnx; skipping sussurro-speak (TTS). See README.")
endif()