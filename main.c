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

    /* skip header */
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file)) {

        double cpu;
        double ram;
        int stress;

        sscanf(line, "%lf,%lf,%d", &cpu, &ram, &stress);

        dataset.cpu_norm[count] = cpu;
        dataset.ram_norm[count] = ram;
        dataset.stress_injected[count] = stress;

        count++;

        if (count >= MAX_DATA)
            break;
    }

    fclose(file);

    dataset.length = count;

    printf("Loaded %d samples\n", count);

    return dataset;
}

int main() {


    data dataset = load_dataset();

    int iterations = 50000;

    printf("Starting training...\n");

    model(dataset, iterations);

    printf("Training complete\n");

    free(dataset.cpu_norm);
    free(dataset.ram_norm);
    free(dataset.stress_injected);

    return 0;
}