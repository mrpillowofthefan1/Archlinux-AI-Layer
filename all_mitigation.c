#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "all_mitigation.h"

void action_none() {
    printf("Nominal. No action required.\n");
}

void get_current_throttle() {
    system("watch -n 1 \"cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq\"");
}

void action_throttle_heaviest_process() {
    system("ps aux --sort=-%cpu | head -n 2 | tail -n 1 | awk '{print $2}' | xargs renice +10 -p");
}



void get_top_processes(char* buffer, int size) {
    FILE* fp = popen("ps -eo pid,pcpu,comm --sort=-pcpu | sed -n '2p'", "r");
    if (fp == NULL) return;
    if (fgets(buffer, size, fp) != NULL)  buffer[strcspn(buffer, "\n")] = 0;
    pclose(fp);
}

void action_kill() {
    char process_name[600];
    char command[500];
    get_top_processes(process_name, sizeof(process_name));
    if (strlen(process_name) > 0) {
        printf("CRITICAL STRESS. Terminating offending processes\n");
        snprintf(command, sizeof(command),"kill -p %s", process_name);
        system(command);
    }
}

void set_state(Action action) {
    switch (action) {
        case 1:
            action_none();
            break;
        case 2:
            printf("Medium level");
            break;
        case 3:
            printf("High level");
            break;
        case 4:
            printf("High level");
            break;
        default:
            action_none();
            break;
    }
}