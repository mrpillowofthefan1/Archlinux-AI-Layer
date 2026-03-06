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
    *a_next_out = a_next; *yt_pred_out = yt_pred;
    cache->a_next = a_next; cache->a_prev = a_prev; cache->xt = xt;
    cache->params = params; cache->y_pred = yt_pred;
    matrix_free(Wax_xt); matrix_free(Waa_aprev); matrix_free(Wya_anext);
}

rnn_forward_output rnn_forward(Matrix** x, int T_x, Matrix* a0, parameter_rnn* params) {
    rnn_forward_output out;
    out.T_x = T_x;
    out.a = e_calloc(T_x, sizeof(Matrix*));
    out.y_pred = e_calloc(T_x, sizeof(Matrix*));
    out.caches = e_calloc(T_x, sizeof(rnn_cache));
    Matrix* a_next = matrix_copy(a0);
    for (int t = 0; t < T_x; t++) {
        rnn_cell_forward(x[t], a_next, params, &out.a[t], &out.y_pred[t], &out.caches[t]);
        matrix_free(a_next);
        a_next = matrix_copy(out.a[t]);
    }
    matrix_free(a_next);
    return out;
}

rnn_gradients rnn_cell_backward(Matrix* da, Matrix* y_true, rnn_cache* cache) {
    rnn_gradients out;
    Matrix* dy = matrix_copy(cache->y_pred);
    matrix_sub(dy, y_true);
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

void lstm_cell_forward(Matrix* xt, Matrix* a_prev, Matrix* c_prev, lstm_parameters* params, Matrix** a_next_out, Matrix** c_next_out, lstm_cache* cache) {
    int n_a = a_prev->rows;
    Matrix* concat = matrix_concat(a_prev, xt);
    Matrix* ft = matrix_zero(n_a, 1);
    Matrix* tmp_f = matrix_mul(params->Wf, concat);
    matrix_add(tmp_f, params->bf);
    sigmoid_forward_matrix(ft, tmp_f);
    Matrix* it = matrix_zero(n_a, 1);
    Matrix* tmp_i = matrix_mul(params->Wi, concat);
    matrix_add(tmp_i, params->bi);
    sigmoid_forward_matrix(it, tmp_i);
    Matrix* cct = matrix_zero(n_a, 1);
    Matrix* tmp_c = matrix_mul(params->Wc, concat);
    matrix_add(tmp_c, params->bc); tanh_forward_matrix(cct, tmp_c);
    Matrix* c_next = matrix_copy(c_prev);
    matrix_hadamard(c_next, ft);
    Matrix* it_cct = matrix_copy(it);
    matrix_hadamard(it_cct, cct);
    matrix_add(c_next, it_cct);
    Matrix* ot = matrix_zero(n_a, 1);
    Matrix* tmp_o = matrix_mul(params->Wo, concat);
    matrix_add(tmp_o, params->bo);
    sigmoid_forward_matrix(ot, tmp_o);
    Matrix* a_next = matrix_zero(n_a, 1);
    tanh_forward_matrix(a_next, c_next);
    matrix_hadamard(a_next, ot);
    *a_next_out = a_next; *c_next_out = c_next;
    cache->a_next = a_next;
    cache->c_next = c_next;
    cache->a_prev = a_prev;
    cache->c_prev = c_prev;
    cache->ft = ft;
    cache->it = it;
    cache->cct = cct;
    cache->ot = ot; cache->xt = xt;
    cache->params = params;
    matrix_free(concat);
    matrix_free(tmp_f);
    matrix_free(tmp_i);
    matrix_free(tmp_c);
    matrix_free(it_cct);
    matrix_free(tmp_o);
}

lstm_forward_output lstm_forward(Matrix** x, int T_x, Matrix* a0, Matrix* c0, lstm_parameters* params) {
    lstm_forward_output out;
    out.T_x = T_x;
    out.a = e_calloc(T_x, sizeof(Matrix*));
    out.y_pred = e_calloc(T_x, sizeof(Matrix*));
    out.caches = e_calloc(T_x, sizeof(lstm_cache));
    Matrix* a_curr = matrix_copy(a0);
    Matrix* c_curr = matrix_copy(c0);
    for (int t = 0; t < T_x; t++) {
        Matrix* c_next;
        lstm_cell_forward(x[t], a_curr, c_curr, params, &out.a[t], &c_next, &out.caches[t]);
        matrix_free(a_curr); matrix_free(c_curr);
        a_curr = matrix_copy(out.a[t]);
        c_curr = matrix_copy(c_next);
        out.y_pred[t] = matrix_mul(params->Wy, out.a[t]);
        matrix_add(out.y_pred[t], params->by);
    }
    matrix_free(a_curr); matrix_free(c_curr);
    return out;
}

lstm_cell_gradients lstm_cell_backward(Matrix *da_next, Matrix *dc_next, lstm_cache *cache) {
    lstm_cell_gradients g;
    int n_a = cache->a_next->rows;
    Matrix* tanh_c = matrix_zero(n_a, 1);
    tanh_forward_matrix(tanh_c, cache->c_next);
    Matrix* dot = matrix_copy(da_next);
    matrix_hadamard(dot, tanh_c);
    Matrix* dtanh_c = matrix_copy(tanh_c);
    for(int i=0; i<n_a; i++) {
        dtanh_c->data[i] = 1.0 - dtanh_c->data[i]*dtanh_c->data[i];
    }
    matrix_hadamard(dtanh_c, cache->ot); matrix_hadamard(dtanh_c, da_next);
    matrix_add(dtanh_c, dc_next);
    Matrix* dcct = matrix_copy(dtanh_c); matrix_hadamard(dcct, cache->it);
    for(int i=0; i<n_a; i++) {
        dcct->data[i] *= (1.0 - cache->cct->data[i]*cache->cct->data[i]);
    }
    Matrix* dit = matrix_copy(dtanh_c); matrix_hadamard(dit, cache->cct);
    for(int i=0; i<n_a; i++) {
        dit->data[i] *= (cache->it->data[i]*(1.0 - cache->it->data[i]));
    }
    Matrix* dft = matrix_copy(dtanh_c); matrix_hadamard(dft, cache->c_prev);
    for(int i=0; i<n_a; i++) {
        dft->data[i] *= (cache->ft->data[i]*(1.0 - cache->ft->data[i]));
    }
    for(int i=0; i<n_a; i++) {
        dot->data[i] *= (cache->ot->data[i]*(1.0 - cache->ot->data[i]));
    }
    Matrix* concat = matrix_concat(cache->a_prev, cache->xt); Matrix* concat_T = matrix_transpose(concat);
    g.dWf = matrix_mul(dft, concat_T); g.dWi = matrix_mul(dit, concat_T);
    g.dWc = matrix_mul(dcct, concat_T); g.dWo = matrix_mul(dot, concat_T);
    g.dbf = matrix_sum_cols(dft); g.dbi = matrix_sum_cols(dit);
    g.dbc = matrix_sum_cols(dcct); g.dbo = matrix_sum_cols(dot);
    Matrix* Wf_a = matrix_slice_cols_start_end(cache->params->Wf, 0, n_a);
    Matrix* Wf_a_T = matrix_transpose(Wf_a); g.da_prev = matrix_mul(Wf_a_T, dft);
    g.dc_prev = matrix_copy(dtanh_c); matrix_hadamard(g.dc_prev, cache->ft);

    matrix_free(tanh_c);
    matrix_free(dot);
    matrix_free(dtanh_c);
    matrix_free(dcct);
    matrix_free(dit);
    matrix_free(dft);
    matrix_free(concat);
    matrix_free(concat_T);
    matrix_free(Wf_a);
    matrix_free(Wf_a_T);
    return g;
}

lstm_gradients lstm_backward(Matrix* da, lstm_cache* cache, int T_x) {
    lstm_gradients out;
    int n_a = da->rows, n_x = cache[0].xt->rows;
    out.dx = matrix_zero(n_x, T_x);
    out.dWf = matrix_zero(n_a, n_a + n_x);
    Matrix* da_prev = matrix_zero(n_a, 1);
    Matrix* dc_prev = matrix_zero(n_a, 1);
    for(int t=T_x-1; t>=0; t--) {
        Matrix* da_t = matrix_slice_col(da, t);
        matrix_add(da_t, da_prev);
        lstm_cell_gradients step = lstm_cell_backward(da_t, dc_prev, &cache[t]);
        matrix_add(out.dWf, step.dWf);
        matrix_free(da_prev); matrix_free(dc_prev);
        da_prev = step.da_prev; dc_prev = step.dc_prev;
        matrix_free(da_t); matrix_free(step.dWf);
    }
    out.da0 = da_prev;
    matrix_free(dc_prev);
    return out;
}