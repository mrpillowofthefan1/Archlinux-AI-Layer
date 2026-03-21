#pragma once


void action_none();
void action_throttle_heaviest_process();
void get_top_processes(char* buffer, int size);
void action_kill();


typedef enum {
    ACTION_NONE,
    ACTION_KILL,
    ACTION_THROTTLE,
    ACTION_OVERCLOCK
} Action;
void set_state(Action action);