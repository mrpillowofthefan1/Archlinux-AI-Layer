#include <stdio.h>
#include <stdlib.h>
#include "train.h"
#include "rnn.h"

#define MAX_DATA 100000

data load_dataset() {
    FILE* file = fopen("/Users/fanman/CLionProjects/Test/processed_system_log.csv", "r");
    if (!file) {
        printf("Error opening dataset\n");
        exit(1);
    }

    data dataset;
    dataset.cpu_norm = malloc(sizeof(double) * MAX_DATA);
    dataset.ram_norm = malloc(sizeof(double) * MAX_DATA);
    dataset.stress_injected = malloc(sizeof(int) * MAX_DATA);

    int count = 0;
    char line[256];
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file)) {
        double cpu, ram;
        int stress;
        if (sscanf(line, "%lf,%lf,%d", &cpu, &ram, &stress) == 3) {
            dataset.cpu_norm[count] = cpu;
            dataset.ram_norm[count] = ram;
            dataset.stress_injected[count] = stress;
            count++;
        }
        if (count >= MAX_DATA) break;
    }
    fclose(file);
    dataset.length = count;
    printf("Loaded %d samples\n", count);
    return dataset;
}

int main() {
    data dataset = load_dataset();
    int iterations = 50000;
    int n_a = 50;
    int n_x = 3;

    printf("Starting training...\n");
    parameter_rnn trained_params = model_trained(dataset, iterations);
    printf("Training complete.\n");
    printf("Starting final evaluation pass...\n");
    int correct_predictions = 0;
    Matrix* eval_a = matrix_zero(n_a, 1);

    for (int i = 0; i < dataset.length; i++) {
        Matrix* xt_eval = matrix_zero(n_x, 1);
        MAT(xt_eval, 0, 0) = dataset.cpu_norm[i];
        MAT(xt_eval, 1, 0) = dataset.ram_norm[i];
        MAT(xt_eval, 2, 0) = (double)dataset.stress_injected[i];

        Matrix* a_next_eval;
        Matrix* yt_pred_eval;
        rnn_cache dummy_cache;

        rnn_cell_forward(xt_eval, eval_a, &trained_params, &a_next_eval, &yt_pred_eval, &dummy_cache);

        double prediction_value = MAT(yt_pred_eval, 0, 0);
        int binary_prediction = (prediction_value > 0.5) ? 1 : 0;

        if (binary_prediction == dataset.stress_injected[i]) {
            correct_predictions++;
        }

        matrix_free(eval_a);
        eval_a = a_next_eval;
        matrix_free(xt_eval);
        matrix_free(yt_pred_eval);
    }

    double final_accuracy = ((double)correct_predictions / dataset.length) * 100.0;
    printf("Final Accuracy: %.2f%% (%d/%d)\n", final_accuracy, correct_predictions, dataset.length);

    matrix_free(eval_a);
    matrix_free(trained_params.Waa);
    matrix_free(trained_params.Wax);
    matrix_free(trained_params.Wya);
    matrix_free(trained_params.ba);
    matrix_free(trained_params.by);

    free(dataset.cpu_norm);
    free(dataset.ram_norm);
    free(dataset.stress_injected);

    return 0;
}