#include "rnn.h"
#include "util.h"

void rnn_cell_forward(Matrix* xt, Matrix* a_prev, parameter_rnn* params, Matrix** a_next_out, Matrix** yt_pred_out, rnn_cache* cache) {
    Matrix* Waa_aprev = matrix_mul(params->Waa, a_prev);
    Matrix* Wax_xt   = matrix_mul(params->Wax, xt);
    matrix_add(Waa_aprev, Wax_xt);
    matrix_add(Waa_aprev, params->ba);

    Matrix* a_next = matrix_zero(Waa_aprev->rows, Waa_aprev->cols);
    tanh_forward_matrix(a_next, Waa_aprev);

    Matrix* Wya_anext = matrix_mul(params->Wya, a_next);
    matrix_add(Wya_anext, params->by);

    Matrix* yt_pred = matrix_zero(Wya_anext->rows, Wya_anext->cols);
    sigmoid_forward_matrix(yt_pred, Wya_anext);

    *a_next_out = a_next;
    *yt_pred_out = yt_pred;


    cache->a_next = a_next;
    cache->a_prev = a_prev;
    cache->xt = xt;
    cache->params = params;
    cache->y_pred = yt_pred;

    matrix_free(Wax_xt);
    matrix_free(Waa_aprev);
    matrix_free(Wya_anext);
}

rnn_forward_output rnn_forward(Matrix** x, int T_x, Matrix* a0, parameter_rnn* params) {
    rnn_forward_output out;
    out.T_x = T_x;
    out.a = e_calloc(T_x, sizeof(Matrix*));
    out.y_pred = e_calloc(T_x, sizeof(Matrix*));
    out.caches = e_calloc(T_x, sizeof(rnn_cache));


    Matrix* a_prev_step = a0;

    for (int t = 0; t < T_x; t++) {
        rnn_cell_forward(x[t], a_prev_step, params, &out.a[t], &out.y_pred[t], &out.caches[t]);
        a_prev_step = out.a[t];
    }

    return out;
}

rnn_gradients rnn_cell_backward(Matrix* da, Matrix* y_true, rnn_cache* cache) {
    rnn_gradients out;
    Matrix* dy = matrix_copy(cache->y_pred);
    matrix_sub(dy, y_true);

    Matrix* Wya_T = matrix_transpose(cache->params->Wya);
    Matrix* da_from_y = matrix_mul(Wya_T, dy);
    matrix_add(da, da_from_y);
    matrix_free(Wya_T);
    matrix_free(da_from_y);

    Matrix* a_next_T = matrix_transpose(cache->a_next);
    out.dWya = matrix_mul(dy, a_next_T);
    out.dby = matrix_sum_cols(dy);

    Matrix* dtanh = matrix_zero(cache->a_next->rows, cache->a_next->cols);
    for (int i = 0; i < dtanh->rows * dtanh->cols; i++)
        dtanh->data[i] = da->data[i] * (1.0 - cache->a_next->data[i] * cache->a_next->data[i]);

    Matrix* Wax_T = matrix_transpose(cache->params->Wax);
    out.dxt = matrix_mul(Wax_T, dtanh);

    Matrix* xt_T = matrix_transpose(cache->xt);
    out.dWax = matrix_mul(dtanh, xt_T);

    Matrix* Waa_T = matrix_transpose(cache->params->Waa);
    out.da_prev = matrix_mul(Waa_T, dtanh);

    Matrix* aprev_T = matrix_transpose(cache->a_prev);
    out.dWaa = matrix_mul(dtanh, aprev_T);

    out.dba = matrix_sum_cols(dtanh);

    matrix_free(dy); matrix_free(a_next_T); matrix_free(dtanh);
    matrix_free(Wax_T); matrix_free(xt_T); matrix_free(Waa_T); matrix_free(aprev_T);

    return out;
}

rnn_gradients rnn_backward(Matrix* da, Matrix** y_true, rnn_cache* caches, int T_x) {
    rnn_gradients out;
    int n_a = da->rows, n_x = caches[0].xt->rows;
    out.dWax = matrix_zero(n_a, n_x); out.dWaa = matrix_zero(n_a, n_a);
    out.dWya = matrix_zero(caches[0].params->Wya->rows, caches[0].params->Wya->cols);
    out.dba = matrix_zero(n_a, 1); out.dby = matrix_zero(out.dWya->rows, 1);
    out.dxt = matrix_zero(n_x, T_x);

    Matrix* da_prev = matrix_zero(n_a, 1);

    for (int t = T_x - 1; t >= 0; t--) {
        Matrix* da_t = matrix_slice_col(da, t);
        matrix_add(da_t, da_prev);

        rnn_gradients step = rnn_cell_backward(da_t, y_true[t], &caches[t]);

        matrix_set_col(out.dxt, t, step.dxt);
        matrix_add(out.dWax, step.dWax);
        matrix_add(out.dWaa, step.dWaa);
        matrix_add(out.dWya, step.dWya);
        matrix_add(out.dba, step.dba);
        matrix_add(out.dby, step.dby);

        matrix_free(da_prev);
        da_prev = step.da_prev;

        matrix_free(da_t);
        matrix_free(step.dxt);
        matrix_free(step.dWax);
        matrix_free(step.dWaa);
        matrix_free(step.dWya);
        matrix_free(step.dba);
        matrix_free(step.dby);
    }

    out.da_prev = da_prev;
    return out;
}