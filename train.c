#include "train.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "rnn.h"
#include "util.h"

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

    Matrix *matrices[] = {dWaa, dWax, dWya, db, dby};
    for (int r = 0; r < 5; r++) {
        matrix_clip(matrices[r], maxValue);
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

optimize_return optimize(
    Matrix** X,
    Matrix** Y,
    int T_x,
    parameter_rnn* params,
    Matrix* a_prev,
    double learning_rate,
    double clip_value
) {
    optimize_return result;

    rnn_forward_output cache = rnn_forward(X, T_x, a_prev, params);

    double loss = 0.0;
    for (int t = 0; t < T_x; t++) {
        for (int i = 0; i < cache.y_pred[t]->rows; i++) {
            double yhat = MAT(cache.y_pred[t], i, 0);
            double y    = MAT(Y[t], i, 0);
            if (y == 1.0)
                loss -= log(yhat + 1e-8);
        }
    }

    Matrix* da = matrix_zero(cache.y_pred[0]->rows, T_x);
    for (int t = 0; t < T_x; t++) {
        Matrix* diff = matrix_copy(cache.y_pred[t]);
        matrix_sub(diff, Y[t]);
        matrix_set_col(da, t, diff);
        matrix_free(diff);
    }

    rnn_gradients grads = rnn_backward(da, NULL, cache.caches, T_x);

    clip_gradients to_clip;
    to_clip.dWaa = grads.dWaa;
    to_clip.dWax = grads.dWax;
    to_clip.dWya = grads.dWya;
    to_clip.db   = grads.dba;
    to_clip.dby  = grads.dby;

    clip(to_clip, clip_value);

    matrix_scalar_mul(to_clip.dWaa, learning_rate);
    matrix_sub(params->Waa, to_clip.dWaa);

    matrix_scalar_mul(to_clip.dWax, learning_rate);
    matrix_sub(params->Wax, to_clip.dWax);

    matrix_scalar_mul(to_clip.dWya, learning_rate);
    matrix_sub(params->Wya, to_clip.dWya);

    matrix_scalar_mul(to_clip.db, learning_rate);
    matrix_sub(params->ba, to_clip.db);

    matrix_scalar_mul(to_clip.dby, learning_rate);
    matrix_sub(params->by, to_clip.dby);

    result.loss = loss;
    result.gradients = grads;
    result.n_a_1 = matrix_copy(cache.a[T_x - 1]);

    matrix_free(da);

    for (int t = 0; t < T_x; t++) {
        matrix_free(cache.a[t]);
        matrix_free(cache.y_pred[t]);
    }

    free(cache.a);
    free(cache.y_pred);
    free(cache.caches);

    return result;
}

double get_initial_loss(const int size, const int len) {
    return -log(1.0/size)*len;
}

parameter_rnn initialize_parameter(int n_a, int n_x, int n_y) {
    parameter_rnn result;
    Matrix* Wax  = matrix_random(n_x, n_a);
    matrix_scalar_mul(Wax, 0.01);
    Matrix* Waa  = matrix_random(n_a, n_a);
    matrix_scalar_mul(Waa, 0.01);
    Matrix* Wya  = matrix_random(n_a, n_y);
    matrix_scalar_mul(Wya, 0.01);
    Matrix* ba = matrix_zero(n_a, 1);
    Matrix* by = matrix_zero(n_y, 1);
    result.Waa = Waa;
    result.Wax = Waa;
    result.Wya = Wya;
    result.ba = ba;
    result.by = by;
    return result;
}

int model(const data data, const int iterations, int n_a, int n_x, int n_y) {
    double cpu_usage = data.cpu_norm;
    double ram_usage = data.ram_norm;
    double stress_injected = data.stress_injected;
    parameter_rnn params = initialize_parameter(n_a, n_x, n_y);
    double loss = get_initial_loss(n_a, n_x);
    for (int t = 0; t < iterations; t++) {
         int idx = t % ;
     }
    return 0;
}


