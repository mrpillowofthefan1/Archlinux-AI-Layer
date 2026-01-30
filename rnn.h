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
    Matrix** a;          // length T_x, each (n_a, m)
    Matrix** y_pred;     // length T_x, each (n_y, m)
    rnn_cache* caches;   // length T_x
    int T_x;
} lstm_forward_output;

