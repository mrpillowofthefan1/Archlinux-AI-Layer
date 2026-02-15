#include <stdio.h>

#include "rnn.h"

void data_preprocessing() {
    FILE* fptr;
    fptr = fopen("processed_system_log.csv", "r");

}

clip_gradients clip(clip_gradients gradients, double maxValue) {
    Matrix* dWaa = gradients.dWaa;
    Matrix* dWax = gradients.dWax;
    Matrix* dWya = gradients.dWya;
    Matrix* db = gradients.db;
    Matrix* dby = gradients.dby;
    clip_gradients gradients2;

    Matrix *all_matricies[] = {dWaa, dWax, dWya, db, dby};
    for (int r = 0; r < 5; r++) {
        matrix_clip(all_matricies[r], maxValue);
    }

    gradients2.dWaa = dWaa;
    gradients2.dWax = dWax;
    gradients2.dWya = dWya;
    gradients2.db = db;
    gradients2.dby = dby;

    return gradients2;
}
