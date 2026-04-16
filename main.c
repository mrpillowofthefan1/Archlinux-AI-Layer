#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <locale.h>
#include "rnn.h"
#include "getResourceUsage.h"
#include "all_mitigation.h"

Matrix* matrix_load_from_file(FILE* fp) {
    int rows, cols;
    if (fread(&rows, sizeof(int), 1, fp) != 1) return NULL;
    if (fread(&cols, sizeof(int), 1, fp) != 1) return NULL;

    Matrix* m = matrix_zero(rows, cols);
    fread(m->data, sizeof(double), rows * cols, fp);
    return m;
}

parameter_rnn load_trained_model(const char* filename) {
    parameter_rnn params;
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("CRITICAL ERROR: Could not find %s. Run your training program first!\n", filename);
        exit(1);
    }

    params.Waa = matrix_load_from_file(fp);
    params.Wax = matrix_load_from_file(fp);
    params.Wya = matrix_load_from_file(fp);
    params.ba  = matrix_load_from_file(fp);
    params.by  = matrix_load_from_file(fp);
    fclose(fp);

    return params;
}

double predict(parameter_rnn* params, Matrix** a_prev_ptr, double cpu, double ram) {
    Matrix* x = matrix_zero(2, 1);
    MAT(x, 0, 0) = cpu;
    MAT(x, 1, 0) = ram;

    Matrix* a_next;
    Matrix* y_pred;
    rnn_cache dummy_cache;

    rnn_cell_forward(x, *a_prev_ptr, params, &a_next, &y_pred, &dummy_cache);

    double result = MAT(y_pred, 0, 0);

    matrix_free(x);
    matrix_free(y_pred);

    matrix_free(*a_prev_ptr);
    *a_prev_ptr = a_next;

    return result;
}

int main() {
    setlocale(LC_NUMERIC, "C");

    printf("Booting AI Stress Mitigation Controller...\n");

    parameter_rnn params = load_trained_model("rnn_weights.bin");
    Matrix* a_prev = matrix_zero(params.Waa->rows, 1);

    printf("Controller Active. Entering 10Hz Mitigation Loop.\n");

    while (1) {
        double cpu_in = get_cpu_usage();
        double ram_in = get_ram_usage();

        double raw_stress = predict(&params, &a_prev, cpu_in, ram_in);

        printf("CPU: %5.2f%% | RAM: %5.2f%% | Stress Prob: %.4f\n", cpu_in * 100.0, ram_in * 100.0, raw_stress);

        Action current_action = ACTION_L0;

        if (raw_stress > 0.8) current_action = ACTION_L3;
        else if (raw_stress > 0.5) current_action = ACTION_L2;
        else if (raw_stress > 0.2) current_action = ACTION_L1;
        else current_action = ACTION_L0;

        set_state(current_action);
        usleep(100000);
    }

    return 0;
}