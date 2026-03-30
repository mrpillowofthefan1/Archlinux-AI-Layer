cmake_minimum_required(VERSION 3.29)
project(Test C)

set(CMAKE_C_STANDARD 23)

# THE DIAGNOSTIC FLAGS:
# -O0 forces zero optimization (stops vectorization bias)
# -g adds debug symbols
# -fno-builtin stops the OS from substituting secret math shortcuts
set(CMAKE_C_FLAGS "-O0 -g -fno-builtin -fno-stack-protector")

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    # Force Linux to store every intermediate float to memory
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffloat-store")
elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    # Stop Mac from fusing multiply and add
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffp-contract=off")
endif()

add_executable(Test main.c
        rnn.c
        rnn.h
        util.c
        util.h
        train.c
        train.h
        getResourceUsage.c
        getResourceUsage.h)

# CRITICAL for Linux math functions!
target_link_libraries(Test PRIVATE m)