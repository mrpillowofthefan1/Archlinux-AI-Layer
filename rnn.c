#include <stddef.h>
#include "rnn.h"

#include "util.h"

double*** rnn_cell_forward(double*** xt, double*** a_prev, parameter parameters) {
    double*** xt_prev = xt;
    double*** hidden= a_prev;
    double*** Wax = parameters.Wax;
    double***  Waa = parameters.Waa;
    double*** Wya = parameters.Wya;
    double*** ba = parameters.ba;
    double*** by = parameters.by;

    double*** a_next = tanh_forward(matrix_scalar_multiply())



}
