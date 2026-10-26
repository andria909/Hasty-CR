/* Optional Android debug-shell transport, built locally, never auto-installed.
 * Requires an administrator to install root:shell with mode 04750 in the guest
 * system image. Android apps cannot execute it. No network or daemon.
 */
#define _GNU_SOURCE
#include <grp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if ((getuid() != 0 && getuid() != 2000) || geteuid() != 0) {
        fputs("hasty-su: requires guest root installation and an ADB shell caller\n", stderr);
        return 126;
    }
    if (argc != 3 || strcmp(argv[1], "-c") || !argv[2][0]) {
        fputs("usage: hasty-su -c COMMAND\n", stderr);
        return 2;
    }
    if (setgroups(0, NULL) || setresgid(0, 0, 0) || setresuid(0, 0, 0)) {
        perror("hasty-su: credentials");
        return 126;
    }
    if (clearenv() || setenv("PATH", "/system/bin:/system/xbin", 1)) {
        perror("hasty-su: environment");
        return 126;
    }
    execl("/system/bin/sh", "sh", "-c", argv[2], (char *)NULL);
    perror("hasty-su: exec");
    return 127;
}
