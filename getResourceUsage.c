#include <stdio.h>


static unsigned long long lastUser, lastNice, lastSystem, lastIdle;
void init_cpu() {
    FILE* fptr = fopen("/proc/stat","r");
    fscanf(fptr,"cpu %llu %llu %llu %llu", &lastUser, &lastNice, &lastSystem, &lastIdle);
    fclose(fptr);
}
