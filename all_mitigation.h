#pragma once

typedef enum {
    ACTION_NONE,
    ACTION_KILL,
    ACTION_THROTTLE,
    ACTION_OVERCLOCK
} Action;


void set_state(Action action);