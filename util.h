#ifndef UTIL_H
#define UTIL_H
#define MAT(M, r, c) ((M)->data[(r) * (M)->cols + (c)])

#include <stddef.h>

void *e_calloc(size_t num, size_t size);

typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;


double randu(void);
double randn(double mean, double std);

Matrix *matrix_zero(int rows, int cols);
Matrix *matrix_random(int rows, int cols);
Matrix *matrix_copy(Matrix *A);
Matrix *matrix_concat(Matrix *A, Matrix *B);
Matrix *matrix_transpose(Matrix *A);
void matrix_free(Matrix *M);
int init_zero_matrix(double*** A, int R, int C);

void matrix_add(Matrix *A, Matrix *B);
void matrix_sub(Matrix *A, Matrix *B);
void matrix_scalar_mul(Matrix *A, double s);
void matrix_hadamard(Matrix *A, Matrix *B);
Matrix *matrix_mul(Matrix *A, Matrix *B);
void matrix_clip(Matrix *A, double limit);

void sigmoid_forward_matrix(Matrix* Y, Matrix* X);
void sigmoid_backward_matrix(Matrix* dldY, Matrix* Y, Matrix* dldX);
void tanh_forward_matrix(Matrix *Y, Matrix *X);
void tanh_backward_matrix(Matrix* dldY, Matrix* Y, Matrix* dldX);

#endif
