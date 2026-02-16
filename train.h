#pragma once
#include "rnn.h"
void data_preprocessing();
clip_gradients clip(clip_gradients gradients, double maxValue);
void clip_test(double mValue);
