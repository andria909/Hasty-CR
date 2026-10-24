/* Read-only diagnostic stream. No ptrace, process writes, taps or network.
 * Layout: official ARM64 160402020, guarded by the Python launcher.
 * Collection address and PID expire with the process/battle; never persist them
 * as global offsets. Coordinates describe current bodies, not cast commands.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_ENTITIES 1024
#define MASK 0x00ffffffffffffffULL
static int memfd;
static uint64_t expected_vtable;
static uint64_t resolved_world;
static int include_own_state;

static uint64_t ns(clockid_t clock_id) {
    struct timespec t;
    clock_gettime(clock_id, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}
static uint64_t q(const void *data, size_t offset) {
    uint64_t v; memcpy(&v, (const char *)data + offset, 8); return v;
}
static int32_t i(const void *data, size_t offset) {
    int32_t v; memcpy(&v, (const char *)data + offset, 4); return v;
}
static int rd(uint64_t address, void *data, size_t size) {
    address &= MASK;
    if (address < 0x10000) return 0;
    return pread(memfd, data, size, (off_t)address) == (ssize_t)size;
}
struct entity {
    uint64_t addr;
    uint8_t raw[0x140];
    int32_t hp, max_hp;
    char name[129];
};
static struct entity units[MAX_ENTITIES];

/* Independently checked on ARM64: private hand against deck-screen order and
 * next-card UI; tick against battle clock; mana against the visible bar.
 * Do not export the opponent's private economy. Existing v6 mode is unchanged.
 */
static void print_own_state(uint64_t collection) {
    uint64_t state, back, players[2], found;
    int32_t tick, end_tick;
    if (!resolved_world || !rd(resolved_world + 0x60, &tick, 4) || tick < 0 || tick > 12000 ||
        !rd(resolved_world + 0xa8, &state, 8) || !rd(state + 8, &back, 8) ||
        (back & MASK) != resolved_world || !rd(resolved_world + 0x40, &found, 8) ||
        (found & MASK) != collection || !rd(state + 0xe0, players, 16)) {
        fputs(",\"own_state\":{\"status\":\"unavailable\"}", stdout); return;
    }
    printf(",\"battle_tick\":%d", tick);
    int matches = 0, side = -1;
    int32_t hand[4], cycle[4], mana = 0;
    uint8_t saved[0x230];
    uint64_t owner = 0;
    for (int n = 0; n < 2; ++n) {
        uint8_t raw[0x230]; int32_t h[4], c[4];
        if (!rd(players[n], raw, sizeof raw) || i(raw, 0x78) != n ||
            (q(raw, 0x10) & MASK) != collection ||
            i(raw, 0x21c) != 4 || i(raw, 0x22c) != 4 ||
            i(raw, 0x218) < 4 || i(raw, 0x218) > 8 ||
            i(raw, 0x228) < 4 || i(raw, 0x228) > 16 ||
            !rd(q(raw, 0x210), h, sizeof h) || !rd(q(raw, 0x220), c, sizeof c)) continue;
        unsigned mask = 0;
        for (int j = 0; j < 8; ++j) {
            int v = j < 4 ? h[j] : c[j - 4];
            if (v < 0 || v > 7 || (mask & (1u << v))) { mask = 0; break; }
            mask |= 1u << v;
        }
        if (mask != 255) continue;
        ++matches; side = n; owner = players[n] & MASK;
        memcpy(hand, h, sizeof h); memcpy(cycle, c, sizeof c);
        memcpy(saved, raw, sizeof saved);
    }
    uint8_t check[0x230]; int32_t h2[4], c2[4];
    if (matches != 1 || !rd(owner + 0x2f8, &mana, 4) || mana < 0 || mana > 100000 ||
        !rd(owner, check, sizeof check) ||
        memcmp(saved + 0x210, check + 0x210, 0x20) ||
        !rd(q(check, 0x210), h2, sizeof h2) || memcmp(hand, h2, sizeof hand) ||
        !rd(q(check, 0x220), c2, sizeof c2) || memcmp(cycle, c2, sizeof cycle) ||
        !rd(resolved_world + 0x60, &end_tick, 4) || end_tick != tick) {
        fputs(",\"own_state\":{\"status\":\"transition_or_ambiguous\"}", stdout); return;
    }
    printf(",\"own_state\":{\"status\":\"ok\",\"world\":\"0x%llx\","
           "\"tick\":%d,\"own_side\":%d,\"elixir_raw\":%d,"
           "\"hand_indices\":[%d,%d,%d,%d],\"cycle_indices\":[%d,%d,%d,%d]}",
           (unsigned long long)resolved_world, tick, side, mana,
           hand[0], hand[1], hand[2], hand[3], cycle[0], cycle[1], cycle[2], cycle[3]);
}

static int snapshot(uint64_t collection, uint64_t sequence) {
    uint8_t h[24], end[24];
    uint64_t pointers[MAX_ENTITIES], final_pointers[MAX_ENTITIES];
    uint64_t start = ns(CLOCK_MONOTONIC), wall = ns(CLOCK_REALTIME);
    if (!rd(collection, h, sizeof h) || q(h, 0) != expected_vtable) return 1;
    int32_t capacity = i(h, 16), count = i(h, 20);
    if (count < 0 || count > capacity || capacity > MAX_ENTITIES) return 2;
    uint64_t array = q(h, 8);
    if (count && !rd(array, pointers, count * 8)) return 3;
    for (int n = 0; n < count; ++n) {
        struct entity *u = &units[n];
        u->addr = pointers[n] & MASK;
        if (!rd(u->addr, u->raw, sizeof u->raw)) return 4;
        if ((q(u->raw, 0x10) & MASK) != collection) return 5;
        for (int k = 0; k < n; ++k) if (units[k].addr == u->addr) return 6;
        uint8_t data[64];
        if (!rd(q(u->raw, 0x48), data, sizeof data)) return 7;
        int32_t length = i(data, 0x2c);
        if (length < 1 || length > 128) return 8;
        if (length < 8) memcpy(u->name, data + 0x30, length);
        else if (!rd(q(data, 0x30), u->name, length)) return 9;
        u->name[length] = 0;
        for (int k = 0; k < length; ++k) {
            char c = u->name[k];
            if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                  (c >= '0' && c <= '9') || c == '_')) return 10;
        }
        u->hp = u->max_hp = -1;
        uint64_t components = q(u->raw, 0x18) & MASK;
        if (components) {
            uint8_t comp[24];
            if (!rd(components, comp, sizeof comp)) return 11;
            uint64_t health = q(comp, 0x10) & MASK;
            if (health) {
                int32_t pair[2];
                if (!rd(health + 0x10, pair, sizeof pair)) return 12;
                u->hp = pair[0]; u->max_hp = pair[1];
            }
        }
        int side = i(u->raw, 0x78), x = i(u->raw, 0x7c), y = i(u->raw, 0x80);
        if (side < 0 || side > 1 || x < -4000 || x > 22000 || y < -4000 || y > 36000)
            return 13;
    }
    if (!rd(collection, end, sizeof end) || memcmp(h, end, sizeof h)) return 14;
    if (count && (!rd(array, final_pointers, count * 8) ||
                  memcmp(pointers, final_pointers, count * 8))) return 15;
    for (int n = 0; n < count; ++n) {
        uint8_t identity[24];
        if (!rd(units[n].addr, identity, sizeof identity) ||
            i(identity, 8) != i(units[n].raw, 8) ||
            q(identity, 16) != q(units[n].raw, 16)) return 16;
    }
    uint64_t end_ns = ns(CLOCK_MONOTONIC);
    printf("{\"sequence\":%llu,\"device_start_ns\":%llu,\"device_end_ns\":%llu,"
           "\"device_wall_ns\":%llu,\"read_us\":%.3f,\"entity_count\":%d,\"entities\":[",
           (unsigned long long)sequence, (unsigned long long)start,
           (unsigned long long)end_ns, (unsigned long long)wall,
           (end_ns - start) / 1000.0, count);
    for (int n = 0; n < count; ++n) {
        struct entity *u = &units[n];
        printf("%s{\"address\":\"0x%llx\",\"entity_id\":%u,\"name\":\"%s\","
               "\"side\":%d,\"x\":%d,\"y\":%d,\"card_id\":%d,\"level_index\":", n ? "," : "",
               (unsigned long long)u->addr, (unsigned)i(u->raw, 8), u->name,
               i(u->raw, 0x78), i(u->raw, 0x7c), i(u->raw, 0x80),
               i(u->raw, 0xac));
        /* Projectile +0x120 is a target coordinate, not a troop level. */
        if ((unsigned)i(u->raw, 8) / 1000000 == 5) printf("%d", i(u->raw, 0x120));
        else printf("null");
        printf(",\"hp\":%d,\"max_hp\":%d", u->hp, u->max_hp);
        printf(",\"deployment_tick_candidate\":%d,\"play_sequence_candidate\":%d,"
               "\"deck_index_candidate\":%d", i(u->raw, 0xa8),
               i(u->raw, 0x98), i(u->raw, 0x94));
        if (!strcmp(u->name, "FireballSpell") || !strcmp(u->name, "ArrowsSpell"))
            printf(",\"target_position_candidate\":{\"x\":%d,\"y\":%d},"
                   "\"source_position_fields\":{\"x\":%d,\"y\":%d}",
                   i(u->raw, 0x120), i(u->raw, 0x124),
                   i(u->raw, 0x12c), i(u->raw, 0x130));
        printf("}");
    }
    fputs("]", stdout);
    if (include_own_state) print_own_state(collection);
    puts("}");
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 6 && !(argc == 7 && (!strcmp(argv[6], "--once") ||
                                      !strcmp(argv[6], "--auto-once") ||
                                      !strcmp(argv[6], "--auto-state-once")))) {
        fputs("usage: arm64_entities PID COLLECTION_OR_LIBG_BASE EXPECTED_VTABLE SECONDS INTERVAL_MS [--once|--auto-once]\n", stderr);
        return 2;
    }
    char path[80];
    char *tail;
    long pid = strtol(argv[1], &tail, 10);
    if (*tail || pid <= 0) return 2;
    snprintf(path, sizeof path, "/proc/%ld/mem", pid);
    uint64_t collection = strtoull(argv[2], &tail, 0) & MASK;
    if (*tail || collection < 0x10000) return 2;
    expected_vtable = strtoull(argv[3], &tail, 0);
    if (*tail || !expected_vtable) return 2;
    int seconds = atoi(argv[4]), interval = atoi(argv[5]);
    if (seconds < 1 || seconds > 600 || interval < 1 || interval > 1000) return 2;
    memfd = open(path, O_RDONLY | O_CLOEXEC);
    if (memfd < 0) { perror("open memory read-only"); return 1; }
    include_own_state = argc == 7 && !strcmp(argv[6], "--auto-state-once");
    if (argc == 7 && (!strcmp(argv[6], "--auto-once") || include_own_state)) {
        /* Pinned build only. Resolve anew on every request, including lobby. */
        uint64_t root, context, world, found, check;
        if (!rd(collection + 0x1a586a8, &root, 8) ||
            !rd((root & MASK) + 0x28, &context, 8) ||
            !rd((context & MASK) + 0x90, &world, 8) ||
            !rd((world & MASK) + 0x40, &found, 8) ||
            !rd(found, &check, 8) || check != expected_vtable) {
            puts("{\"status\":\"no_battle\"}"); close(memfd); return 4;
        }
        collection = found & MASK;
        resolved_world = world & MASK;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    uint64_t stop = ns(CLOCK_MONOTONIC) + (uint64_t)seconds * 1000000000;
    for (uint64_t sequence = 0; ns(CLOCK_MONOTONIC) < stop; ++sequence) {
        int code = snapshot(collection, sequence);
        if (code) printf("{\"sequence\":%llu,\"rejected\":%d,\"device_wall_ns\":%llu}\n",
                         (unsigned long long)sequence, code,
                         (unsigned long long)ns(CLOCK_REALTIME));
        if (argc == 7) { close(memfd); return code ? 3 : 0; }
        struct timespec pause = {interval / 1000, interval % 1000 * 1000000L};
        nanosleep(&pause, NULL);
    }
    close(memfd);
    return 0;
}
