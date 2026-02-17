#pragma once
#include "rnn.h"

typedef struct {
    double loss;
    rnn_gradients gradients;
    Matrix* n_a_1;
}optimize_return;

typedef struct {
    double cpu_norm;
    double ram_norm;
    int stress_injected;
}data;

void data_preprocessing();
clip_gradients clip(clip_gradients gradients, double maxValue);
void clip_test(double mValue);
