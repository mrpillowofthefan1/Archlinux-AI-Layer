#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "all_mitigation.h"

# define CPU_MAX_PATH "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
# define CPU_MIN_PATH "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq"
# define CPU_CUR_PATH "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
# define CPU_TEMP_PATH "/sys/class/thermal/thermal_zone0/temp"

void action_none() {
    printf("Nominal. No action required.\n");
}

unsigned long read_freq(const char* path) {
    unsigned long freq = 0;
    FILE* fp = fopen(path, "r");
    if (fp) {
        if (fscanf(fp, "%lu", &freq) != 1) freq = 0;
        fclose(fp);
    }
    return freq;
}

unsigned long read_temp() {
    unsigned long temp = 0;
    FILE* fp = fopen(CPU_TEMP_PATH, "r");
    if (fp) {
        if (fscanf(fp, "%lu", &temp) != 1) temp = 0;
        fclose(fp);
    }
    return temp/1000;
}

void set_max_throttle() {
    unsigned long max_val = read_freq(CPU_MAX_PATH);
    if (max_val > 0) {
        system("for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -f \"$i\" ] && echo performance > \"$i\"; done 2>/dev/null");
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do [ -f \"$i\" ] && echo %lu > \"$i\"; done 2>/dev/null", max_val);
        system(cmd);
    }
}

void set_min_throttle() {
    unsigned long min_val = read_freq(CPU_MIN_PATH);
    if (min_val > 0) {
        system("for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -f \"$i\" ] && echo powersave > \"$i\"; done 2>/dev/null");
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do [ -f \"$i\" ] && echo %lu > \"$i\"; done 2>/dev/null", min_val);
        system(cmd);
    }
}

void action_reset_throttle() {
    const unsigned long min_val = read_freq(CPU_MIN_PATH);
    const unsigned long max_val = read_freq(CPU_MAX_PATH);

    if (min_val > 0 && max_val > 0) {
        printf("RESTORING: Resetting CPU to standard dynamic scaling.\n");
        char cmd_min[256], cmd_max[256];
        snprintf(cmd_min, sizeof(cmd_min), "for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do [ -f \"$i\" ] && echo %lu > \"$i\"; done 2>/dev/null", min_val);
        snprintf(cmd_max, sizeof(cmd_max), "for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do [ -f \"$i\" ] && echo %lu > \"$i\"; done 2>/dev/null", max_val);

        system(cmd_min);
        system(cmd_max);

        system("for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -f \"$i\" ] && echo schedutil > \"$i\"; done 2>/dev/null");
    }
}

void action_throttle_heaviest_process() {
    system("ps -N -u root -o pid= --sort=-%cpu | head -n 1 | xargs -r renice +10 -p > /dev/null 2>&1");
}

void get_top_pid(char* buffer, int size) {
    FILE* fp = popen("ps -N -u root -o pid= --sort=-%cpu | head -n 1 | tr -d ' '", "r");
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
        snprintf(command, sizeof(command), "kill -9 %s", pid);
        system(command);
    }
}

void action_flush_ram() {
    system("sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null");
}

void set_state(Action action) {
    switch (action) {
        case 0:
            action_none();
            set_min_throttle();
            break;
        case 1:
            action_throttle_heaviest_process();
            set_min_throttle();
            break;
        case 2:
            action_kill();
            action_flush_ram();
            action_reset_throttle();
            break;
        case 3:
            action_kill();
            action_throttle_heaviest_process();
            action_flush_ram();
            if (read_temp()<90) set_max_throttle();
            break;
        default:
            action_none();
            set_min_throttle();
            break;
    }
}