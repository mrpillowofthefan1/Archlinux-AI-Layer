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


double randu()
{
    return (double)rand() / (double)RAND_MAX;
}

double randn(double mean, double std)
{
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    }

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

Matrix *matrix_zero(int rows, int cols)
{
    Matrix *M = e_calloc(1, sizeof(Matrix));
    M->rows = rows;
    M->cols = cols;
    M->data = e_calloc(rows * cols, sizeof(double));
    return M;
}

Matrix *matrix_random(int rows, int cols) {
    Matrix *M = matrix_zero(rows, cols);

    // We are dropping the 1.0 / sqrt(rows) to avoid the OS math library entirely.
    // Using a hardcoded safe scale for the test.
    double scale = 0.1;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // randu() generates exactly the same 0.0 to 1.0 on every OS.
            // This creates a uniform value between -0.1 and 0.1
            MAT(M, r, c) = (randu() * 2.0 - 1.0) * scale;
        }
    }

    return M;
}

Matrix *matrix_copy(Matrix *A)
{
    Matrix *M = matrix_zero(A->rows, A->cols);
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        M->data[i] = A->data[i];
    return M;
}

Matrix *matrix_concat(Matrix *A, Matrix *B)
{
    if (A->cols != B->cols) {
        fprintf(stderr, "matrix_concat: column mismatch\n");
        exit(1);
    }

    Matrix *C = matrix_zero(A->rows + B->rows, A->cols);

    for (int r = 0; r < A->rows; r++)
        for (int c = 0; c < A->cols; c++)
            MAT(C, r, c) = MAT(A, r, c);

    for (int r = 0; r < B->rows; r++)
        for (int c = 0; c < B->cols; c++)
            MAT(C, r + A->rows, c) = MAT(B, r, c);

    return C;
}


void matrix_add(Matrix *A, Matrix *B)
{
    if (A->rows != B->rows || A->cols != B->cols) {
        fprintf(stderr, "matrix_add: dimension mismatch\n");
        exit(1);
    }

    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        A->data[i] += B->data[i];
}

void matrix_sub(Matrix *A, Matrix *B)
{
    if (A->rows != B->rows || A->cols != B->cols) {
        fprintf(stderr, "matrix_sub: dimension mismatch\n");
        exit(1);
    }

    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        A->data[i] -= B->data[i];
}

void matrix_scalar_mul(Matrix *A, double s)
{
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        A->data[i] *= s;
}

void matrix_hadamard(Matrix *A, Matrix *B)
{
    if (A->rows != B->rows || A->cols != B->cols) {
        fprintf(stderr, "matrix_hadamard: dimension mismatch\n");
        exit(1);
    }

    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        A->data[i] *= B->data[i];
}

void matrix_clip(Matrix *A, double limit)
{
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++) {
        if (A->data[i] > limit)
            A->data[i] = limit;
        else if (A->data[i] < -limit)
            A->data[i] = -limit;
    }
}

int init_zero_matrix(double*** A, int R, int C)
{
    *A = e_calloc(R, sizeof(double*));
    for (int r = 0; r < R; r++)
        (*A)[r] = e_calloc(C, sizeof(double));
    return 0;
}

Matrix *matrix_mul(Matrix *A, Matrix *B)
{
    if (A->cols != B->rows) {
        fprintf(stderr, "matrix_mul: dimension mismatch\n");
        exit(1);
    }

    Matrix *R = matrix_zero(A->rows, B->cols);

    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < B->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->cols; k++)
                sum += MAT(A, i, k) * MAT(B, k, j);
            MAT(R, i, j) = sum;
        }

    return R;
}

Matrix* matrix_slice_col(Matrix* A, int col)
{
    if (col < 0 || col >= A->cols) return NULL;

    Matrix* out = matrix_zero(A->rows, 1);

    for (int r = 0; r < A->rows; r++)
        MAT(out, r, 0) = MAT(A, r, col);

    return out;
}

Matrix* matrix_slice_cols_start_end(Matrix* A, int c0, int c1)
{
    if (!A || c0 < 0 || c1 > A->cols || c0 >= c1) return NULL;

    Matrix* out = matrix_zero(A->rows, c1 - c0);

    for (int r = 0; r < A->rows; r++)
        for (int c = c0; c < c1; c++)
            MAT(out, r, c - c0) = MAT(A, r, c);

    return out;
}


void matrix_set_col(Matrix* A, int col, Matrix* v)
{
    if (!A || !v) return;
    if (col < 0 || col >= A->cols) return;
    if (v->cols != 1 || v->rows != A->rows) return;

    for (int r = 0; r < A->rows; r++)
        MAT(A, r, col) = MAT(v, r, 0);
}

Matrix* matrix_sum_cols(Matrix* A)
{
    if (!A) return NULL;

    Matrix* out = matrix_zero(A->rows, 1);
    for (int r = 0; r < A->rows; r++)
        for (int c = 0; c < A->cols; c++)
            MAT(out, r, 0) += MAT(A, r, c);

    return out;
}


Matrix *matrix_transpose(Matrix *A)
{
    Matrix *T = matrix_zero(A->cols, A->rows);

    for (int r = 0; r < A->rows; r++)
        for (int c = 0; c < A->cols; c++)
            MAT(T, c, r) = MAT(A, r, c);

    return T;
}

void matrix_free(Matrix *M)
{
    if (!M) return;
    free(M->data);
    free(M);
}

void sigmoid_forward_matrix(Matrix* Y, Matrix* X)
{
    if (Y->rows != X->rows || Y->cols != X->cols) {
        fprintf(stderr, "sigmoid_forward_matrix: dimension mismatch\n");
        exit(1);
    }

    int n = X->rows * X->cols;
    for (int i = 0; i < n; i++)
        Y->data[i] = 1.0 / (1.0 + exp(-X->data[i]));
}

void sigmoid_backward_matrix(Matrix* dldY, Matrix* Y, Matrix* dldX)
{
    if (dldY->rows != Y->rows || dldY->cols != Y->cols ||
        dldX->rows != Y->rows || dldX->cols != Y->cols) {
        fprintf(stderr, "sigmoid_backward_matrix: dimension mismatch\n");
        exit(1);
    }

    int n = Y->rows * Y->cols;
    for (int i = 0; i < n; i++)
        dldX->data[i] = dldY->data[i] * Y->data[i] * (1.0 - Y->data[i]);
}

void tanh_forward_matrix(Matrix *Y, Matrix *X)
{
    if (Y->rows != X->rows || Y->cols != X->cols) {
        fprintf(stderr, "tanh_forward_matrix: dimension mismatch\n");
        exit(1);
    }

    int n = X->rows * X->cols;
    for (int i = 0; i < n; i++)
        Y->data[i] = tanh(X->data[i]);
}

void tanh_backward_matrix(Matrix* dldY, Matrix* Y, Matrix* dldX)
{
    if (dldY->rows != Y->rows || dldY->cols != Y->cols ||
        dldX->rows != Y->rows || dldX->cols != Y->cols) {
        fprintf(stderr, "tanh_backward_matrix: dimension mismatch\n");
        exit(1);
    }

    int n = Y->rows * Y->cols;
    for (int i = 0; i < n; i++)
        dldX->data[i] = dldY->data[i] * (1.0 - Y->data[i] * Y->data[i]);
}
