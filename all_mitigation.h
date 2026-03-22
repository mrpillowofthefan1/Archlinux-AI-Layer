#pragma once

typedef enum {
    ACTION_L0,
    ACTION_L1,
    ACTION_L2,
    ACTION_L3,
    ACTION_DEFAULT
} Action;


void set_state(Action action);