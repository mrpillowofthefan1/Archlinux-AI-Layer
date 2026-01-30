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
            x[t],      // input at time t
            a_next,    // previous hidden state
            c_next,    // previous cell state
            params,    // LSTM parameters
            &a_t,      // output hidden state
            &c_t,      // output cell state
            &y_t,      // output prediction
            &out.caches[t] // store cache for backprop
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