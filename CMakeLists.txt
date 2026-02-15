cmake_minimum_required(VERSION 3.29)
project(Test C)

set(CMAKE_C_STANDARD 23)

add_executable(Test main.c
        rnn.c
        rnn.h
        util.c
        util.h
        train.c
        train.h)
