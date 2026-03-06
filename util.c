#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"

void* e_calloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (!ptr) {
        fprintf(stderr, "calloc failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

double randu() { return (double)rand() / (double)RAND_MAX; }

double randn(double mean, double std) {
    static int has_spare = 0;
    static double spare;
    if (has_spare) { has_spare = 0; return mean + std * spare; }
    has_spare = 1;
    double u, v, s;
    do {
        u = randu() * 2.0 - 1.0;
        v = randu() * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

Matrix *matrix_zero(int rows, int cols) {
    Matrix *M = e_calloc(1, sizeof(Matrix));
    M->rows = rows; M->cols = cols;
    M->data = e_calloc(rows * cols, sizeof(double));
    return M;
}

Matrix *matrix_random(int rows, int cols) {
    Matrix *M = matrix_zero(rows, cols);
    double scale = 1.0 / sqrt(rows);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            MAT(M, r, c) = randn(0.0, 1.0) * scale;
    return M;
}

Matrix *matrix_copy(Matrix *A) {
    Matrix *M = matrix_zero(A->rows, A->cols);
    for (int i = 0; i < A->rows * A->cols; i++) M->data[i] = A->data[i];
    return M;
}

void matrix_add(Matrix *A, Matrix *B) {
    for (int i = 0; i < A->rows * A->cols; i++) A->data[i] += B->data[i];
}

void matrix_sub(Matrix *A, Matrix *B) {
    for (int i = 0; i < A->rows * A->cols; i++) A->data[i] -= B->data[i];
}

void matrix_scalar_mul(Matrix *A, double s) {
    for (int i = 0; i < A->rows * A->cols; i++) A->data[i] *= s;
}

void matrix_hadamard(Matrix *A, Matrix *B) {
    for (int i = 0; i < A->rows * A->cols; i++) A->data[i] *= B->data[i];
}

void matrix_clip(Matrix *A, double limit) {
    for (int i = 0; i < A->rows * A->cols; i++) {
        if (A->data[i] > limit) A->data[i] = limit;
        else if (A->data[i] < -limit) A->data[i] = -limit;
    }
}

Matrix *matrix_mul(Matrix *A, Matrix *B) {
    Matrix *R = matrix_zero(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < B->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->cols; k++) sum += MAT(A, i, k) * MAT(B, k, j);
            MAT(R, i, j) = sum;
        }
    return R;
}

Matrix* matrix_slice_col(Matrix* A, int col) {
    Matrix* out = matrix_zero(A->rows, 1);
    for (int r = 0; r < A->rows; r++) MAT(out, r, 0) = MAT(A, r, col);
    return out;
}

void matrix_set_col(Matrix* A, int col, Matrix* v) {
    for (int r = 0; r < A->rows; r++) MAT(A, r, col) = MAT(v, r, 0);
}

Matrix* matrix_sum_cols(Matrix* A) {
    Matrix* out = matrix_zero(A->rows, 1);
    for (int r = 0; r < A->rows; r++)
        for (int c = 0; c < A->cols; c++) MAT(out, r, 0) += MAT(A, r, c);
    return out;
}

Matrix *matrix_transpose(Matrix *A) {
    Matrix *T = matrix_zero(A->cols, A->rows);
    for (int r = 0; r < A->rows; r++)
        for (int c = 0; c < A->cols; c++) MAT(T, c, r) = MAT(A, r, c);
    return T;
}

void matrix_free(Matrix *M) { if (M) { free(M->data); free(M); } }

void sigmoid_forward_matrix(Matrix* Y, Matrix* X) {
    for (int i = 0; i < X->rows * X->cols; i++) Y->data[i] = 1.0 / (1.0 + exp(-X->data[i]));
}

void tanh_forward_matrix(Matrix *Y, Matrix *X) {
    for (int i = 0; i < X->rows * X->cols; i++) Y->data[i] = tanh(X->data[i]);
}

Matrix *matrix_concat(Matrix *A, Matrix *B) {
    Matrix *C = matrix_zero(A->rows + B->rows, A->cols);
    for (int r = 0; r < A->rows; r++) MAT(C, r, 0) = MAT(A, r, 0);
    for (int r = 0; r < B->rows; r++) MAT(C, r + A->rows, 0) = MAT(B, r, 0);
    return C;
}

Matrix* matrix_slice_cols_start_end(Matrix* A, int c0, int c1) {
    Matrix* out = matrix_zero(A->rows, c1 - c0);
    for (int r = 0; r < A->rows; r++)
        for (int c = c0; c < c1; c++) MAT(out, r, c - c0) = MAT(A, r, c);
    return out;
}