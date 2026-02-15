#pragma once
#include "util.h"

typedef struct {
    Matrix* Waa;
    Matrix* Wax;
    Matrix* Wya;
    Matrix* ba;
    Matrix* by;

}parameter_rnn;

typedef struct {
    Matrix *Wf, *bf;
    Matrix *Wi, *bi;
    Matrix *Wc, *bc;
    Matrix *Wo, *bo;
    Matrix *Wy, *by;
} lstm_parameters;

typedef struct {
    Matrix *a_next;
    Matrix *c_next;
    Matrix *a_prev;
    Matrix *c_prev;
    Matrix *ft;
    Matrix *it;
    Matrix *cct;
    Matrix *ot;
    Matrix *xt;
    lstm_parameters *params;
} lstm_cache;


typedef struct {
    Matrix* a_next;
    Matrix* a_prev;
    Matrix* xt;
    parameter_rnn* params;
} rnn_cache;


typedef struct {
    Matrix** a;          // length T_x, each (n_a, m)
    Matrix** y_pred;     // length T_x, each (n_y, m)
    rnn_cache* caches;   // length T_x
    int T_x;
} rnn_forward_output;

typedef struct {
    int T_x;
    Matrix** a;
    Matrix** y_pred;
    lstm_cache* caches;
} lstm_forward_output;

typedef struct {
    Matrix* dxt;
    Matrix* da_prev;
    Matrix* dWax;
    Matrix* dWaa;
    Matrix* dba;

}rnn_gradients;

typedef struct {
    Matrix *dxt;
    Matrix *da_prev;
    Matrix *dc_prev;
    Matrix *dWf, *dWi, *dWc, *dWo;
    Matrix *dbf, *dbi, *dbc, *dbo;
} lstm_cell_gradients;

typedef struct {
    Matrix* dx;
    Matrix* da0;

    Matrix* dWf;
    Matrix* dWi;
    Matrix* dWc;
    Matrix* dWo;

    Matrix* dbf;
    Matrix* dbi;
    Matrix* dbc;
    Matrix* dbo;
} lstm_gradients;

typedef struct {
    Matrix* dWaa;
    Matrix* dWax;
    Matrix* dWya;
    Matrix* db;
    Matrix* dby;

}clip_gradients;






