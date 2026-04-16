#define GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>

#include <dirent.h>
#include <ctype.h>
#include <time.h>
#include <sys/resource.h>
#include "all_mitigation.h"

static time_t last_affinity_time = 0;

int is_numeric(const char* str) {
    while (*str) {
        if (!isdigit(*str)) return 0;
        str++;
    }
    return 1;
}

int is_stress_process(pid_t pid) {
    char path[256], comm[256];
    snprintf(path, sizeof(path), "/proc/%d/comm", pid);

    FILE *fp = fopen(path, "r");
    if (!fp) return 0;

    if (fgets(comm, sizeof(comm), fp)) {
        comm[strcspn(comm, "\n")] = 0;
        fclose(fp);
        if (strncmp(comm, "stress", 6) == 0) return 1;
    } else {
        fclose(fp);
    }
    return 0;
}

void apply_mitigation_state(Action state) {
    time_t now = time(NULL);
    if (now - last_affinity_time < 15) return;
    last_affinity_time = now;

    DIR *dir = opendir("/proc");
    if (!dir) return;

    struct dirent *ent;
    int current_core = 0;
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);

    while ((ent = readdir(dir)) != NULL) {
        if (is_numeric(ent->d_name)) {
            pid_t pid = atoi(ent->d_name);

            if (is_stress_process(pid)) {
                cpu_set_t mask;
                CPU_ZERO(&mask);

                switch (state) {
                    case ACTION_L0:
                        for (int i = 0; i < num_cores; i++) {
                            CPU_SET(i, &mask);
                        }
                        setpriority(PRIO_PROCESS, pid, 0);
                        break;

                    case ACTION_L1:
                        CPU_SET(current_core, &mask);
                        setpriority(PRIO_PROCESS, pid, 0);
                        current_core++;
                        if (current_core >= num_cores) current_core = 0;
                        break;

                    case ACTION_L2:
                        CPU_SET(current_core, &mask);
                        setpriority(PRIO_PROCESS, pid, 10);
                        current_core++;
                        if (current_core >= num_cores) current_core = 0;
                        break;

                    case ACTION_L3:
                        CPU_SET(current_core, &mask);
                        setpriority(PRIO_PROCESS, pid, 19);
                        current_core++;
                        if (current_core >= num_cores) current_core = 0;
                        break;

                    default:
                        break;
                }
                sched_setaffinity(pid, sizeof(cpu_set_t), &mask);
            }
        }
    }
    closedir(dir);
}

void set_state(Action action) {
    apply_mitigation_state(action);
}