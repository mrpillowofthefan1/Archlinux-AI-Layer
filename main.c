#include <stdio.h>
#include <stdlib.h>
#include "train.h"

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

double predict(parameter_rnn* params, Matrix* a_prev, double cpu, double ram) {
    Matrix* x = matrix_zero(2, 1);

    MAT(x, 0, 0) = cpu;
    MAT(x, 1, 0) = ram;

    Matrix* a_next;
    Matrix* y_pred;
    rnn_cache cache;

    rnn_cell_forward(x, a_prev, params, &a_next, &y_pred, &cache);

    double result = MAT(y_pred, 0, 0);

    matrix_free(x);
    matrix_free(a_next);
    matrix_free(y_pred);

    return result;
}

int main() {
    data dataset = load_dataset();

    int iterations = 50000;

    printf("Starting training...\n");

    parameter_rnn params = model(dataset, iterations);

    printf("Training complete\n");

    Matrix* a_prev = matrix_zero(50, 1); // hidden size

    while (1) {
        double cpu, ram;

        printf("\nEnter CPU (0-1): ");
        if (scanf("%lf", &cpu) != 1) break; // Added basic check to allow exiting cleanly on bad input

        printf("Enter RAM (0-1): ");
        if (scanf("%lf", &ram) != 1) break;

        // Get the raw probability from the model
        double raw_stress = predict(&params, a_prev, cpu, ram);

        // Threshold the output: 1 if >= 0.5, else 0
        int final_stress = (raw_stress >= 0.5) ? 1 : 0;

        // Print both the binary class and the raw probability
        printf("Predicted Stress: %d (Raw confidence: %.4f)\n", final_stress, raw_stress);
    }

    free(dataset.cpu_norm);
    free(dataset.ram_norm);
    free(dataset.stress_injected);

    return 0;
}