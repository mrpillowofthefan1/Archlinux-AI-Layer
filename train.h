#pragma once
#include "rnn.h"

typedef struct {
    double loss;
    rnn_gradients gradients;
    Matrix* n_a_1;
}optimize_return;
void data_preprocessing();
clip_gradients clip(clip_gradients gradients, double maxValue);
void clip_test(double mValue);
