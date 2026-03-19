#define _GNU_SOURCE
#include <stdio.h>
#include <sys/sysinfo.h>


static unsigned long long lastUser, lastNice, lastSystem, lastIdle;

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

double get_ram_usage() {
    struct sysinfo info;
    sysinfo(&info);

    long long total = info.totalram * info.mem_unit;
    long long free  = info.freeram * info.mem_unit;

    return ((double)(total - free) / total) * 100.0;
}

