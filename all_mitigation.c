#include <stdio.h>
#include <stdlib.h>

void action_none() {
    printf("[AI] Status: Nominal. No action required.\n");
}

void action_throttle() {
    printf("[AI] Status: High CPU detected. Throttling background processes\n");
    system("pkill -STOP -f 'heavy_task_name'");
}

void action_kill() {
    printf("[AI] Status: CRITICAL STRESS. Terminating offending processes\n");
    system("pkill -9 -f 'heavy_task_name'");
}