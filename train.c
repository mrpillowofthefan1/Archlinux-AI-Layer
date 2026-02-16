#include <math.h>
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


void clip_test(double mValue) {

    clip_gradients gradients;

    Matrix* dWax = matrix_random(5, 3);
    matrix_scalar_mul(dWax, 10);

    Matrix* dWaa = matrix_random(5, 5);
    matrix_scalar_mul(dWaa, 10);

    Matrix* dWya = matrix_random(2, 5);
    matrix_scalar_mul(dWya, 10);

    Matrix* db = matrix_random(5, 1);
    matrix_scalar_mul(db, 10);

    Matrix* dby = matrix_random(2, 1);
    matrix_scalar_mul(dby, 10);

    gradients.dWax = dWax;
    gradients.dWaa = dWaa;
    gradients.dWya = dWya;
    gradients.db   = db;
    gradients.dby  = dby;

    clip_gradients gradients_processed = clip(gradients, mValue);

    Matrix* original[] = {
        gradients.dWax,
        gradients.dWaa,
        gradients.dWya,
        gradients.db,
        gradients.dby
    };

    Matrix* clipped[] = {
        gradients_processed.dWax,
        gradients_processed.dWaa,
        gradients_processed.dWya,
        gradients_processed.db,
        gradients_processed.dby
    };

    const char* names[] = {
        "dWax", "dWaa", "dWya", "db", "dby"
    };

    for (int m = 0; m < 5; m++) {

        Matrix* valuei = original[m];
        Matrix* valuef = clipped[m];

        for (int i = 0; i < valuef->rows; i++) {
            for (int j = 0; j < valuef->cols; j++) {

                int idx = i * valuef->cols + j;

                double orig  = valuei->data[idx];
                double clipv = valuef->data[idx];

                if (clipv < -fabs(mValue)) {
                    printf("Problem with %s: value below -mValue\n", names[m]);
                }

                if (clipv > fabs(mValue)) {
                    printf("Problem with %s: value above mValue\n", names[m]);
                }

                if (orig <= mValue && orig >= -mValue) {
                    if (orig != clipv) {
                        printf("Problem with %s: value changed incorrectly\n", names[m]);
                    }
                }
            }
        }

    }

    printf("Clip test complete.\n");
}
