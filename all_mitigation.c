#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "all_mitigation.h"


# define  CPU_MAX_PATH "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
# define CPU_MIN_PATH "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq"
# define CPU_CUR_PATH "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"


void action_none() {
    printf("Nominal. No action required.\n");
}

void get_current_throttle() {
    system("watch -n 1 \"cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq\"");
}

void set_max_throttle() {
    system("for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee $i; done");
}

void set_min_throttle() {
    system("for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo powersave | sudo tee $i; done");
}



void action_throttle_heaviest_process() {
    system("ps aux --sort=-%cpu | head -n 2 | tail -n 1 | awk '{print $2}' | xargs renice +10 -p");
}



void get_top_pid(char* buffer, int size) {
    FILE* fp = popen("ps -eo pid,ni --sort=-pcpu | awk '$2 > 0 {print $1}' | head -n 1", "r");
    if (fp == NULL) return;
    if (fgets(buffer, size, fp) != NULL) {
        buffer[strcspn(buffer, "\n")] = 0;
    }
    pclose(fp);
}

void action_kill() {
    char pid[32];
    char command[64];
    get_top_pid(pid, sizeof(pid));
    if (strlen(pid) > 0) {
        printf("CRITICAL STRESS. Terminating PID: %s\n", pid);
        snprintf(command, sizeof(command), "kill -9 %s", pid);
        system(command);
    }
}

void set_state(Action action) {
    switch (action) {
        case 0:
            action_none();
            break;
        case 1:
            action_throttle_heaviest_process();
            break;
        case 2:
            action_kill();
            break;
        case 3:
            action_kill();
            action_throttle_heaviest_process();
            action_throttle_heaviest_process();
            break;
        default:
            action_none();
            break;
    }
}