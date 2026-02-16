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

void clip_test() {
    Matrix* dWax = matrix_random(5,3);
    matrix_scalar_mul(dWax,10);
    Matrix* dWaa = matrix_random(5,3);
    matrix_scalar_mul(dWaa,10);
    Matrix* dWya = matrix_random(5,3);
    matrix_scalar_mul(dWya,10);
    Matrix* db = matrix_random(5,3);
    matrix_scalar_mul(db,10);
    Matrix* dby = matrix_random(5,3);
    matrix_scalar_mul(dby,10);
}