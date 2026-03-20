#include "train.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "rnn.h"
#include "util.h"

#define SEQ_LEN 10

clip_gradients clip(clip_gradients g, double maxValue) {
    matrix_clip(g.dWaa, maxValue);
    matrix_clip(g.dWax, maxValue);
    matrix_clip(g.dWya, maxValue);
    matrix_clip(g.db, maxValue);
    matrix_clip(g.dby, maxValue);
    return g;
}

parameter_rnn initialize_parameter(int n_a, int n_x, int n_y) {
    parameter_rnn res;
    res.Wax = matrix_random(n_a, n_x);
    res.Waa = matrix_random(n_a, n_a);
    res.Wya = matrix_random(n_y, n_a);
    res.ba = matrix_zero(n_a, 1);
    res.by = matrix_zero(n_y, 1);
    return res;
}

optimize_return optimize(Matrix** X, Matrix** Y, int T_x, parameter_rnn* params, Matrix* a_prev, double lr, double clip_val) {
    optimize_return result;

    rnn_forward_output cache = rnn_forward(X, T_x, a_prev, params);

    double loss = 0.0;
    Matrix* da = matrix_zero(params->Waa->rows, T_x);

    for (int t = 0; t < T_x; t++) {

        double yhat = MAT(cache.y_pred[t], 0, 0);
        double y    = MAT(Y[t], 0, 0);

        loss -= (y * log(yhat + 1e-8));

        Matrix* diff = matrix_copy(cache.y_pred[t]);
        matrix_sub(diff, Y[t]);

        matrix_set_col(da, t, diff);
        matrix_free(diff);
    }

    rnn_gradients grads = rnn_backward(da, Y, cache.caches, T_x);

    clip_gradients to_clip = {grads.dWaa, grads.dWax, grads.dWya, grads.dba, grads.dby};
    clip_gradients clipped_grads = clip(to_clip, clip_val);

    matrix_scalar_mul(clipped_grads.dWaa, lr); matrix_sub(params->Waa, clipped_grads.dWaa);
    matrix_scalar_mul(clipped_grads.dWax, lr); matrix_sub(params->Wax, clipped_grads.dWax);
    matrix_scalar_mul(clipped_grads.dWya, lr); matrix_sub(params->Wya, clipped_grads.dWya);
    matrix_scalar_mul(clipped_grads.db, lr);  matrix_sub(params->ba, clipped_grads.db);
    matrix_scalar_mul(clipped_grads.dby, lr);  matrix_sub(params->by, clipped_grads.dby);

    rnn_gradients final_grads = {grads.dxt, grads.da_prev, clipped_grads.dWax,clipped_grads.dWaa, clipped_grads.dWya,clipped_grads.db, clipped_grads.dby};
    result.loss = loss;
    result.gradients = final_grads;
    result.n_a_1 = matrix_copy(cache.a[T_x - 1]);

    matrix_free(da);

    matrix_free(grads.dxt);
    matrix_free(grads.dWax);
    matrix_free(grads.dWaa);
    matrix_free(grads.dWya);
    matrix_free(grads.dba);
    matrix_free(grads.dby);
    matrix_free(grads.da_prev);
    matrix_free(final_grads.dxt);
    matrix_free(final_grads.dWax);
    matrix_free(final_grads.dWaa);
    matrix_free(final_grads.dWya);
    matrix_free(final_grads.dba);
    matrix_free(final_grads.dby);
    matrix_free(final_grads.da_prev);
    matrix_free(clipped_grads.dWax);
    matrix_free(clipped_grads.dWaa);
    matrix_free(clipped_grads.dWya);
    matrix_free(clipped_grads.dby);

    for (int t = 0; t < T_x; t++) {
        matrix_free(cache.a[t]);
        matrix_free(cache.y_pred[t]);
    }

    free(cache.a);
    free(cache.y_pred);
    free(cache.caches);

    return result;
}

parameter_rnn model(const data data_feed, const int iterations) {
    const int n_a = 50, n_x = 2, n_y = 1;

    parameter_rnn params = initialize_parameter(n_a, n_x, n_y);

    Matrix* a_prev = matrix_zero(n_a, 1);

    double loss = -log(1.0 / n_y);

    for (int t = 0; t < iterations; t++) {

        int idx = t % data_feed.length;

        Matrix** X = malloc(sizeof(Matrix*));
        Matrix** Y = malloc(sizeof(Matrix*));

        X[0] = matrix_zero(n_x, 1);
        MAT(X[0], 0, 0) = data_feed.cpu_norm[idx];
        MAT(X[0], 1, 0) = data_feed.ram_norm[idx];

        Y[0] = matrix_zero(n_y, 1);
        MAT(Y[0], 0, 0) = (double)data_feed.stress_injected[idx];

        optimize_return res = optimize(
            X,
            Y,
            1,
            &params,
            a_prev,
            0.01,
            5.0
        );

        loss = 0.999 * loss + 0.001 * res.loss;

        matrix_free(a_prev);
        a_prev = res.n_a_1;

        if (t % 1000 == 0) {
            printf("Iteration %d, Loss %f\n", t, loss);
        }

        matrix_free(X[0]);
        matrix_free(Y[0]);
        free(X);
        free(Y);
    }

    matrix_free(a_prev);

    return params;
}