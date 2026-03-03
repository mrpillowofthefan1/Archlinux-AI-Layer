#include <stdio.h>


static unsigned long long lastUser, lastNice, lastSystem, lastIdle;

//TODO: implement the functions that constantly get system information at that time
void init_cpu() {
    FILE* fptr = fopen("/proc/stat","r");
    fscanf(fptr,"cpu %llu %llu %llu %llu", &lastUser, &lastNice, &lastSystem, &lastIdle);
    fclose(fptr);
}

double get_cpu_usage() {
    FILE *fp = fopen("/proc/stat", "r");
    unsigned long long user, nice, system, idle;
    fscanf(fp, "cpu %llu %llu %llu %llu",
           &user, &nice, &system, &idle);
    fclose(fp);

    unsigned long long prevIdle = lastIdle;
    unsigned long long prevTotal = lastUser + lastNice + lastSystem + lastIdle;

    unsigned long long idleDiff = idle - prevIdle;
    unsigned long long totalDiff =
        (user + nice + system + idle) - prevTotal;

    lastUser = user;
    lastNice = nice;
    lastSystem = system;
    lastIdle = idle;

    if (totalDiff == 0) return 0.0;
    return (1.0 - ((double)idleDiff / totalDiff)) * 100.0;
}
