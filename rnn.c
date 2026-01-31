#include "rnn.h"
#include "util.h"

void rnn_cell_forward(
    Matrix* xt,
    Matrix* a_prev,
    parameter_rnn* params,
    Matrix** a_next_out,
    Matrix** yt_pred_out,
    rnn_cache* cache
)
{
    Matrix* Waa_aprev = matrix_mul(params->Waa, a_prev);
    Matrix* Wax_xt   = matrix_mul(params->Wax, xt);
    matrix_add(Waa_aprev, Wax_xt);        // Waa*a_prev + Wax*xt
    matrix_add(Waa_aprev, params->ba);    // + ba

    Matrix* a_next = matrix_zero(Waa_aprev->rows, Waa_aprev->cols);
    tanh_forward_matrix(a_next, Waa_aprev); // a_next = tanh(Waa*a_prev + Wax*xt + ba)

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

    matrix_free(Wax_xt);
    matrix_free(Waa_aprev);
    matrix_free(Wya_anext);
}

rnn_forward_output rnn_forward(Matrix** x, int T_x, Matrix* a0, parameter_rnn* params)
{
    rnn_forward_output out;
    out.T_x = T_x;

    out.a = e_calloc(T_x, sizeof(Matrix*));
    out.y_pred = e_calloc(T_x, sizeof(Matrix*));
    out.caches = e_calloc(T_x, sizeof(rnn_cache));

    Matrix* a_next = matrix_copy(a0);

    for (int t = 0; t < T_x; t++) {
        Matrix* a_t;
        Matrix* y_t;

        rnn_cell_forward(
            x[t],
            a_next,
            params,
            &a_t,
            &y_t,
            &out.caches[t]
        );

        out.a[t] = a_t;
        out.y_pred[t] = y_t;

        if (t > 0)
            matrix_free(a_next);

        a_next = matrix_copy(a_t);
    }

    matrix_free(a_next);

    return out;
}

void lstm_cell_forward(
    Matrix *xt,
    Matrix *a_prev,
    Matrix *c_prev,
    lstm_parameters *params,
    Matrix **a_next_out,
    Matrix **c_next_out,
    Matrix **yt_pred_out,
    lstm_cache *cache
)
{
    Matrix *concat = matrix_concat(a_prev, xt);

    Matrix *ft_lin = matrix_mul(params->Wf, concat);
    matrix_add(ft_lin, params->bf);
    Matrix *ft = matrix_zero(ft_lin->rows, ft_lin->cols);
    sigmoid_forward_matrix(ft, ft_lin);

    Matrix *it_lin = matrix_mul(params->Wi, concat);
    matrix_add(it_lin, params->bi);
    Matrix *it = matrix_zero(it_lin->rows, it_lin->cols);
    sigmoid_forward_matrix(it, it_lin);

    Matrix *cct_lin = matrix_mul(params->Wc, concat);
    matrix_add(cct_lin, params->bc);
    Matrix *cct = matrix_zero(cct_lin->rows, cct_lin->cols);
    tanh_forward_matrix(cct, cct_lin);

    Matrix *c_next = matrix_copy(c_prev);
    matrix_hadamard(c_next, ft);

    Matrix *it_cct = matrix_copy(cct);
    matrix_hadamard(it_cct, it);
    matrix_add(c_next, it_cct);

    Matrix *ot_lin = matrix_mul(params->Wo, concat);
    matrix_add(ot_lin, params->bo);
    Matrix *ot = matrix_zero(ot_lin->rows, ot_lin->cols);
    sigmoid_forward_matrix(ot, ot_lin);

    Matrix *tanh_c = matrix_zero(c_next->rows, c_next->cols);
    tanh_forward_matrix(tanh_c, c_next);

    Matrix *a_next = matrix_copy(tanh_c);
    matrix_hadamard(a_next, ot);

    Matrix *y_lin = matrix_mul(params->Wy, a_next);
    matrix_add(y_lin, params->by);
    Matrix *yt_pred = matrix_zero(y_lin->rows, y_lin->cols);
    sigmoid_forward_matrix(yt_pred, y_lin);

    *a_next_out = a_next;
    *c_next_out = c_next;
    *yt_pred_out = yt_pred;

    cache->a_next = a_next;
    cache->c_next = c_next;
    cache->a_prev = a_prev;
    cache->c_prev = c_prev;
    cache->ft = ft;
    cache->it = it;
    cache->cct = cct;
    cache->ot = ot;
    cache->xt = xt;
    cache->params = params;

    matrix_free(concat);
    matrix_free(ft_lin);
    matrix_free(it_lin);
    matrix_free(cct_lin);
    matrix_free(it_cct);
    matrix_free(ot_lin);
    matrix_free(tanh_c);
    matrix_free(y_lin);
}

lstm_forward_output lstm_forward(Matrix** x, int T_x, Matrix* a0, lstm_parameters* params)
{
    lstm_forward_output out;
    out.T_x = T_x;

    out.a = e_calloc(T_x, sizeof(Matrix*));
    out.y_pred = e_calloc(T_x, sizeof(Matrix*));
    out.caches = e_calloc(T_x, sizeof(lstm_cache));

    Matrix* a_next = matrix_copy(a0);
    Matrix* c_next = matrix_zero(a0->rows, a0->cols);

    for (int t = 0; t < T_x; t++)
    {
        Matrix* a_t;
        Matrix* c_t;
        Matrix* y_t;

        lstm_cell_forward(
            x[t],
            a_next,
            c_next,
            params,
            &a_t,
            &c_t,
            &y_t,
            &out.caches[t]
        );

        out.a[t] = a_t;
        out.y_pred[t] = y_t;

        if (t > 0)
        {
            matrix_free(a_next);
            matrix_free(c_next);
        }

        a_next = matrix_copy(a_t);
        c_next = matrix_copy(c_t);

        matrix_free(c_t);
    }

    matrix_free(a_next);
    matrix_free(c_next);

    return out;
}

rnn_gradients rnn_cell_backward(Matrix* da, rnn_cache* cache) {
    Matrix* a_next = cache->a_next;
    Matrix* a_prev = cache->a_prev;
    Matrix* xt     = cache->xt;
    parameter_rnn* params = cache->params;
    rnn_gradients out;

    Matrix* dtanh = matrix_copy(a_next);
    for (int i = 0; i < a_next->rows * a_next->cols; i++)
        dtanh->data[i] = da->data[i] * (1.0 - a_next->data[i] * a_next->data[i]);

    Matrix* dxt = matrix_mul(matrix_transpose(params->Wax), dtanh);
    Matrix* dWax = matrix_mul(dtanh, matrix_transpose(xt));
    Matrix* da_prev = matrix_mul(matrix_transpose(params->Waa), dtanh);
    Matrix* dWaa = matrix_mul(dtanh, matrix_transpose(a_prev));

    Matrix* dba = matrix_zero(dtanh->rows, 1);
    for (int r = 0; r < dtanh->rows; r++)
        for (int c = 0; c < dtanh->cols; c++)
            MAT(dba, r, 0) += MAT(dtanh, r, c);

    out.dxt = dxt;
    out.dWax = dWax;
    out.da_prev = da_prev;
    out.dWaa = dWaa;
    out.dba = dba;

    matrix_free(dtanh);
    return out;
}

rnn_gradients rnn_backward(Matrix* da, rnn_cache* caches, int T_x) {
    rnn_gradients out;

    int n_a = da->rows;
    int m   = da->cols / T_x;
    int n_x = caches[0].xt->rows;

    Matrix* dx   = matrix_zero(n_x, m * T_x);
    Matrix* dWax = matrix_zero(n_a, n_x);
    Matrix* dWaa = matrix_zero(n_a, n_a);
    Matrix* dba  = matrix_zero(n_a, 1);
    Matrix* da_prev = matrix_zero(n_a, m);

    for (int t = T_x - 1; t >= 0; t--) {
        Matrix* da_t = matrix_slice_col(da, t);
        matrix_add(da_t, da_prev);

        rnn_gradients step = rnn_cell_backward(da_t, &caches[t]);

        matrix_set_col(dx, t, step.dxt);
        matrix_add(dWax, step.dWax);
        matrix_add(dWaa, step.dWaa);
        matrix_add(dba, step.dba);

        matrix_free(da_prev);
        da_prev = step.da_prev;

        matrix_free(step.dxt);
        matrix_free(step.dWax);
        matrix_free(step.dWaa);
        matrix_free(step.dba);
        matrix_free(da_t);
    }

    out.dxt = dx;
    out.dWax = dWax;
    out.dWaa = dWaa;
    out.dba = dba;
    out.da_prev = da_prev;

    return out;
}

lstm_cell_gradients lstm_cell_backward(Matrix *da_next, Matrix *dc_next, lstm_cache *cache) {
    lstm_cell_gradients g;

    Matrix *a_next = cache->a_next;
    Matrix *c_next = cache->c_next;
    Matrix *a_prev = cache->a_prev;
    Matrix *c_prev = cache->c_prev;
    Matrix *ft = cache->ft;
    Matrix *it = cache->it;
    Matrix *cct = cache->cct;
    Matrix *ot = cache->ot;
    Matrix *xt = cache->xt;
    lstm_parameters *p = cache->params;

    int n_a = a_next->rows;
    int n_x = xt->rows;
    int m   = xt->cols;

    Matrix *tanh_c = matrix_zero(n_a, m);
    tanh_forward_matrix(tanh_c, c_next);

    Matrix *dot = matrix_copy(da_next);
    matrix_hadamard(dot, tanh_c);
    matrix_hadamard(dot, ot);

    Matrix *dtanh_c = matrix_copy(tanh_c);
    for (int i = 0; i < dtanh_c->rows * dtanh_c->cols; i++)
        dtanh_c->data[i] = 1.0 - dtanh_c->data[i] * dtanh_c->data[i];
    matrix_hadamard(dtanh_c, ot);
    matrix_hadamard(dtanh_c, da_next);

    Matrix *dcct = matrix_copy(dc_next);
    matrix_hadamard(dcct, it);
    Matrix *tmp = matrix_copy(dtanh_c);
    matrix_hadamard(tmp, it);
    matrix_add(dcct, tmp);
    for (int i = 0; i < dcct->rows * dcct->cols; i++)
        dcct->data[i] *= (1.0 - cct->data[i] * cct->data[i]);

    Matrix *dit = matrix_copy(dc_next);
    matrix_hadamard(dit, cct);
    tmp = matrix_copy(dtanh_c);
    matrix_hadamard(tmp, cct);
    matrix_add(dit, tmp);
    matrix_hadamard(dit, it);
    for (int i = 0; i < dit->rows * dit->cols; i++)
        dit->data[i] *= (1.0 - it->data[i]);

    Matrix *dft = matrix_copy(dc_next);
    matrix_hadamard(dft, c_prev);
    tmp = matrix_copy(dtanh_c);
    matrix_hadamard(tmp, c_prev);
    matrix_add(dft, tmp);
    matrix_hadamard(dft, ft);
    for (int i = 0; i < dft->rows * dft->cols; i++)
        dft->data[i] *= (1.0 - ft->data[i]);

    Matrix *concat = matrix_concat(a_prev, xt);
    Matrix *concat_T = matrix_transpose(concat);

    g.dWf = matrix_mul(dft, concat_T);
    g.dWi = matrix_mul(dit, concat_T);
    g.dWc = matrix_mul(dcct, concat_T);
    g.dWo = matrix_mul(dot, concat_T);

    g.dbf = matrix_sum_cols(dft);
    g.dbi = matrix_sum_cols(dit);
    g.dbc = matrix_sum_cols(dcct);
    g.dbo = matrix_sum_cols(dot);

    Matrix *Wf_a = matrix_slice_cols_start_end(p->Wf, 0, n_a);
    Matrix *Wi_a = matrix_slice_cols_start_end(p->Wi, 0, n_a);
    Matrix *Wc_a = matrix_slice_cols_start_end(p->Wc, 0, n_a);
    Matrix *Wo_a = matrix_slice_cols_start_end(p->Wo, 0, n_a);

    g.da_prev = matrix_mul(matrix_transpose(Wf_a), dft);
    matrix_add(g.da_prev, matrix_mul(matrix_transpose(Wi_a), dit));
    matrix_add(g.da_prev, matrix_mul(matrix_transpose(Wc_a), dcct));
    matrix_add(g.da_prev, matrix_mul(matrix_transpose(Wo_a), dot));

    g.dc_prev = matrix_copy(dc_next);
    matrix_hadamard(g.dc_prev, ft);
    tmp = matrix_copy(dtanh_c);
    matrix_hadamard(tmp, ft);
    matrix_add(g.dc_prev, tmp);

    Matrix *Wf_x = matrix_slice_cols_start_end(p->Wf, n_a, n_a + n_x);
    Matrix *Wi_x = matrix_slice_cols_start_end(p->Wi, n_a, n_a + n_x);
    Matrix *Wc_x = matrix_slice_cols_start_end(p->Wc, n_a, n_a + n_x);
    Matrix *Wo_x = matrix_slice_cols_start_end(p->Wo, n_a, n_a + n_x);

    g.dxt = matrix_mul(matrix_transpose(Wf_x), dft);
    matrix_add(g.dxt, matrix_mul(matrix_transpose(Wi_x), dit));
    matrix_add(g.dxt, matrix_mul(matrix_transpose(Wc_x), dcct));
    matrix_add(g.dxt, matrix_mul(matrix_transpose(Wo_x), dot));

    matrix_free(tanh_c);
    matrix_free(dot);
    matrix_free(dtanh_c);
    matrix_free(tmp);
    matrix_free(concat);
    matrix_free(concat_T);
    matrix_free(Wf_a);
    matrix_free(Wi_a);
    matrix_free(Wc_a);
    matrix_free(Wo_a);
    matrix_free(Wf_x);
    matrix_free(Wi_x);
    matrix_free(Wc_x);
    matrix_free(Wo_x);

    return g;
}

lstm_gradients lstm_backward(Matrix* da, lstm_cache* cache, int T_x)
{
    lstm_gradients out;

    int n_a = da->rows;
    int m   = da->cols / T_x;
    int n_x = cache[0].xt->rows;

    Matrix* dx  = matrix_zero(n_x, m * T_x);
    Matrix* da0 = matrix_zero(n_a, m);

    Matrix* dWf = matrix_zero(n_a, n_a + n_x);
    Matrix* dWi = matrix_zero(n_a, n_a + n_x);
    Matrix* dWc = matrix_zero(n_a, n_a + n_x);
    Matrix* dWo = matrix_zero(n_a, n_a + n_x);

    Matrix* dbf = matrix_zero(n_a, 1);
    Matrix* dbi = matrix_zero(n_a, 1);
    Matrix* dbc = matrix_zero(n_a, 1);
    Matrix* dbo = matrix_zero(n_a, 1);

    Matrix* da_prevt = matrix_zero(n_a, m);
    Matrix* dc_prevt = matrix_zero(n_a, m);

    for (int t = T_x - 1; t >= 0; t--) {
        Matrix* da_t = matrix_slice_col(da, t);
        matrix_add(da_t, da_prevt);

        lstm_cell_gradients step = lstm_cell_backward(da_t, dc_prevt, &cache[t]);

        matrix_set_col(dx, t, step.dxt);

        matrix_add(dWf, step.dWf);
        matrix_add(dWi, step.dWi);
        matrix_add(dWc, step.dWc);
        matrix_add(dWo, step.dWo);

        matrix_add(dbf, step.dbf);
        matrix_add(dbi, step.dbi);
        matrix_add(dbc, step.dbc);
        matrix_add(dbo, step.dbo);

        matrix_free(da_prevt);
        matrix_free(dc_prevt);

        da_prevt = step.da_prev;
        dc_prevt = step.dc_prev;

        matrix_free(da_t);
        matrix_free(step.dxt);
        matrix_free(step.dWf);
        matrix_free(step.dWi);
        matrix_free(step.dWc);
        matrix_free(step.dWo);
        matrix_free(step.dbf);
        matrix_free(step.dbi);
        matrix_free(step.dbc);
        matrix_free(step.dbo);
    }

    da0 = matrix_copy(da_prevt);

    out.dx  = dx;
    out.da0 = da0;
    out.dWf = dWf;
    out.dWi = dWi;
    out.dWc = dWc;
    out.dWo = dWo;
    out.dbf = dbf;
    out.dbi = dbi;
    out.dbc = dbc;
    out.dbo = dbo;

    matrix_free(da_prevt);
    matrix_free(dc_prevt);

    return out;
}




