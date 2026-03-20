#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void action_none() {
    printf("Nominal. No action required.\n");
}

void get_current_throttle() {
    system("watch -n 1 \"cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq\"");
}

void action_throttle() {
    printf("High CPU detected. Throttling background processes\n");
    system("pkill -STOP -f 'heavy_task_name'");
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