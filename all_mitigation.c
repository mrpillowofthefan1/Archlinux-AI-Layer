#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include "all_mitigation.h"

void get_top_pid(char* buffer, int size) {

    FILE* fp = popen("pgrep -x stress-ng | head -n 1", "r");
    if (fp && fgets(buffer, size, fp) && strlen(buffer) > 1) {
        buffer[strcspn(buffer, "\n")] = 0;
        pclose(fp);
        return;
    }
    if (fp) pclose(fp);

    fp = popen("pgrep -x stress | head -n 1", "r");
    if (fp && fgets(buffer, size, fp) && strlen(buffer) > 1) {
        buffer[strcspn(buffer, "\n")] = 0;
        pclose(fp);
        return;
    }
    if (fp) pclose(fp);

    const char* critical_ui = "(gnome|kde|Xorg|wayland|dbus|systemd|pulse|pipewire|bash|zsh|ssh|python|test)";
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "ps -eo pid,comm --sort=-%%cpu | "
        "awk -v mypid=%d '"
        "$1 == mypid || $1 == 1 {next} "
        "$2 ~ /%s/ {next} "
        "{print $1}' | head -n 1",
        getpid(), critical_ui);

    fp = popen(cmd, "r");
    if (fp) {
        if (fgets(buffer, size, fp)) {
            buffer[strcspn(buffer, "\n")] = 0;
        }
        pclose(fp);
    }
}

void set_state(Action action) {
    char pid_str[32] = {0};
    get_top_pid(pid_str, sizeof(pid_str));

    if (strlen(pid_str) == 0) return;
    pid_t pid = atoi(pid_str);
    if (pid <= 1) return;

    pid_t pgid = getpgid(pid);
    if (pgid <= 1) pgid = pid;

    switch (action) {
        case ACTION_L0:
            killpg(pgid, SIGCONT);
            break;

        case ACTION_L1:
            killpg(pgid, SIGCONT);
            usleep(150000);
            killpg(pgid, SIGSTOP);
            break;

        case ACTION_L2:
            killpg(pgid, SIGCONT);
            usleep(42000);
            killpg(pgid, SIGSTOP);
            break;

        case ACTION_L3:
            killpg(pgid, SIGCONT);
            usleep(5000);
            killpg(pgid, SIGSTOP);
            break;

        default:
            break;
    }
}