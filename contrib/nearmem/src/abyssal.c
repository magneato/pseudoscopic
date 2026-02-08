/*
 * abyssal.c - Deep-Sea Bioluminescent Circuit Debugger Implementation
 *
 * "In darkness, we find clarity. In silence, we hear the signals."
 *
 * Copyright (C) 2025 Neural Splines LLC
 * License: MIT
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <signal.h>
#include <stdarg.h>

#include "abyssal.h"

/*
 * ════════════════════════════════════════════════════════════════════════════
 * TERMINAL CONTROL
 * ════════════════════════════════════════════════════════════════════════════
 */

static struct termios orig_termios;
static bool terminal_raw = false;

static void terminal_restore(void) {
    if (terminal_raw) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        printf("\033[?25h");  /* Show cursor */
        printf("\033[?1049l"); /* Exit alternate screen */
        fflush(stdout);
        terminal_raw = false;
    }
}

static void terminal_raw_mode(void) {
    if (!terminal_raw) {
        tcgetattr(STDIN_FILENO, &orig_termios);
        atexit(terminal_restore);
        
        struct termios raw = orig_termios;
        raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
        raw.c_oflag &= ~(OPOST);
        raw.c_cflag |= (CS8);
        raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 1;
        
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        printf("\033[?1049h"); /* Enter alternate screen */
        printf("\033[?25l");   /* Hide cursor */
        fflush(stdout);
        terminal_raw = true;
    }
}

static void get_terminal_size(int *width, int *height) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        *width = ws.ws_col;
        *height = ws.ws_row;
    } else {
        *width = 80;
        *height = 24;
    }
}

static void cursor_move(int row, int col) {
    printf("\033[%d;%dH", row, col);
}

static void clear_screen(void) {
    printf(BG_ABYSS "\033[2J\033[H");
}

static void clear_line(void) {
    printf("\033[2K");
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ════════════════════════════════════════════════════════════════════════════
 */

int abyssal_init(abyssal_ctx_t *ctx, gpufpga_ctx_t *fpga) {
    if (!ctx)
        return -1;
    
    memset(ctx, 0, sizeof(*ctx));
    ctx->fpga = fpga;
    
    /* Allocate waveform storage */
    ctx->waveforms.max_signals = 256;
    ctx->waveforms.signals = calloc(ctx->waveforms.max_signals, 
                                     sizeof(waveform_signal_t));
    if (!ctx->waveforms.signals)
        return -1;
    
    /* Allocate fault storage */
    ctx->waveforms.max_faults = 1024;
    ctx->waveforms.faults = calloc(ctx->waveforms.max_faults,
                                    sizeof(fault_record_t));
    
    /* Allocate breakpoint storage */
    ctx->max_breakpoints = 64;
    ctx->breakpoints = calloc(ctx->max_breakpoints, sizeof(breakpoint_t));
    
    /* Default view settings */
    ctx->view_mode = VIEW_WAVEFORM;
    ctx->zoom_level = 1;
    ctx->signal_pane_width = 20;
    ctx->info_pane_height = 6;
    
    get_terminal_size(&ctx->term_width, &ctx->term_height);
    ctx->waveform_pane_width = ctx->term_width - ctx->signal_pane_width - 3;
    
    strcpy(ctx->status_msg, "Welcome to Abyssal. Press '?' for help.");
    
    return 0;
}

void abyssal_shutdown(abyssal_ctx_t *ctx) {
    if (!ctx)
        return;
    
    /* Free waveform data */
    for (uint32_t i = 0; i < ctx->waveforms.num_signals; i++) {
        free(ctx->waveforms.signals[i].changes);
    }
    free(ctx->waveforms.signals);
    free(ctx->waveforms.faults);
    free(ctx->breakpoints);
    
    if (ctx->active_trace) {
        abyssal_free_trace(ctx->active_trace);
    }
    
    terminal_restore();
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * WAVEFORM MANAGEMENT
 * ════════════════════════════════════════════════════════════════════════════
 */

int abyssal_add_signal(abyssal_ctx_t *ctx, wire_id_t wire, const char *name) {
    if (!ctx || ctx->waveforms.num_signals >= ctx->waveforms.max_signals)
        return -1;
    
    waveform_signal_t *sig = &ctx->waveforms.signals[ctx->waveforms.num_signals];
    
    sig->wire = wire;
    strncpy(sig->name, name, sizeof(sig->name) - 1);
    sig->is_clock = (strstr(name, "clk") != NULL || strstr(name, "CLK") != NULL);
    sig->is_bus = false;
    sig->bus_width = 1;
    
    /* Allocate change history */
    sig->max_changes = 4096;
    sig->changes = malloc(sig->max_changes * sizeof(*sig->changes));
    sig->num_changes = 0;
    
    ctx->waveforms.num_signals++;
    return ctx->waveforms.num_signals - 1;
}

int abyssal_capture_cycle(abyssal_ctx_t *ctx) {
    if (!ctx || !ctx->fpga)
        return -1;
    
    uint64_t cycle = ctx->fpga->cycles_simulated;
    
    for (uint32_t i = 0; i < ctx->waveforms.num_signals; i++) {
        waveform_signal_t *sig = &ctx->waveforms.signals[i];
        
        /* Get current value */
        uint8_t raw_val = gpufpga_get_wire(ctx->fpga, sig->wire);
        sig_value_t val = raw_val ? SIG_1 : SIG_0;
        
        /* Check if changed from last capture */
        bool changed = true;
        if (sig->num_changes > 0) {
            if (sig->changes[sig->num_changes - 1].value == val) {
                changed = false;
            }
        }
        
        if (changed) {
            /* Record the change */
            if (sig->num_changes < sig->max_changes) {
                sig->changes[sig->num_changes].cycle = cycle;
                sig->changes[sig->num_changes].value = val;
                sig->num_changes++;
                sig->num_transitions++;
            }
            
            /* Glitch detection: changed twice in same cycle */
            if (sig->num_changes >= 2) {
                uint32_t prev = sig->num_changes - 2;
                if (sig->changes[prev].cycle == cycle) {
                    sig->num_glitches++;
                    
                    /* Record fault */
                    if (ctx->waveforms.num_faults < ctx->waveforms.max_faults) {
                        fault_record_t *f = &ctx->waveforms.faults[ctx->waveforms.num_faults++];
                        f->type = FAULT_GLITCH;
                        f->severity = SEVERITY_WARNING;
                        f->cycle = cycle;
                        f->wire = sig->wire;
                        snprintf(f->message, sizeof(f->message),
                                "Glitch on signal '%s' at cycle %lu", sig->name, cycle);
                    }
                }
            }
        }
        
        /* Update duty cycle stats */
        if (val == SIG_1) sig->time_high++;
        else sig->time_low++;
    }
    
    ctx->waveforms.end_cycle = cycle;
    ctx->waveforms.total_cycles = cycle - ctx->waveforms.start_cycle + 1;
    
    return 0;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * SIGNAL VALUE LOOKUP
 * ════════════════════════════════════════════════════════════════════════════
 */

sig_value_t abyssal_get_signal_at(abyssal_ctx_t *ctx, wire_id_t wire, uint64_t cycle) {
    /* Find signal in our waveform database */
    for (uint32_t i = 0; i < ctx->waveforms.num_signals; i++) {
        waveform_signal_t *sig = &ctx->waveforms.signals[i];
        if (sig->wire != wire)
            continue;
        
        /* Binary search for the value at this cycle */
        if (sig->num_changes == 0)
            return SIG_X;
        
        /* Find last change at or before this cycle */
        uint32_t lo = 0, hi = sig->num_changes;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (sig->changes[mid].cycle <= cycle)
                lo = mid + 1;
            else
                hi = mid;
        }
        
        if (lo == 0)
            return SIG_X;  /* Before first change */
        
        return sig->changes[lo - 1].value;
    }
    
    return SIG_X;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * SIMULATION CONTROL
 * ════════════════════════════════════════════════════════════════════════════
 */

int abyssal_step(abyssal_ctx_t *ctx) {
    if (!ctx || !ctx->fpga)
        return -1;
    
    gpufpga_step(ctx->fpga);
    abyssal_capture_cycle(ctx);
    ctx->cursor_cycle = ctx->fpga->cycles_simulated;
    
    return 0;
}

int abyssal_run_to(abyssal_ctx_t *ctx, uint64_t target_cycle) {
    if (!ctx || !ctx->fpga)
        return -1;
    
    while (ctx->fpga->cycles_simulated < target_cycle) {
        gpufpga_step(ctx->fpga);
        abyssal_capture_cycle(ctx);
        
        /* Check breakpoints */
        for (uint32_t i = 0; i < ctx->num_breakpoints; i++) {
            if (!ctx->breakpoints[i].enabled)
                continue;
            
            if (ctx->breakpoints[i].trigger_cycle > 0 &&
                ctx->fpga->cycles_simulated >= ctx->breakpoints[i].trigger_cycle) {
                ctx->breakpoints[i].hit = true;
                ctx->cursor_cycle = ctx->fpga->cycles_simulated;
                snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                        "Breakpoint hit at cycle %lu", ctx->fpga->cycles_simulated);
                return 1;
            }
            
            sig_value_t val = abyssal_get_signal_at(ctx, ctx->breakpoints[i].wire,
                                                     ctx->fpga->cycles_simulated);
            if (val == ctx->breakpoints[i].trigger_value) {
                ctx->breakpoints[i].hit = true;
                ctx->cursor_cycle = ctx->fpga->cycles_simulated;
                snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                        "Breakpoint hit: signal condition at cycle %lu",
                        ctx->fpga->cycles_simulated);
                return 1;
            }
        }
    }
    
    ctx->cursor_cycle = ctx->fpga->cycles_simulated;
    return 0;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * TIMELINE NAVIGATION (SCRUBBING)
 * ════════════════════════════════════════════════════════════════════════════
 */

void abyssal_goto_cycle(abyssal_ctx_t *ctx, uint64_t cycle) {
    if (!ctx)
        return;
    
    if (cycle > ctx->waveforms.end_cycle)
        cycle = ctx->waveforms.end_cycle;
    if (cycle < ctx->waveforms.start_cycle)
        cycle = ctx->waveforms.start_cycle;
    
    ctx->cursor_cycle = cycle;
    
    /* Adjust view window if cursor is outside */
    if (cycle < ctx->view_start_cycle) {
        ctx->view_start_cycle = cycle;
        ctx->view_end_cycle = cycle + ctx->waveform_pane_width * ctx->zoom_level;
    } else if (cycle > ctx->view_end_cycle) {
        ctx->view_end_cycle = cycle;
        ctx->view_start_cycle = cycle - ctx->waveform_pane_width * ctx->zoom_level;
        if ((int64_t)ctx->view_start_cycle < 0)
            ctx->view_start_cycle = 0;
    }
}

void abyssal_seek_forward(abyssal_ctx_t *ctx, int64_t delta) {
    abyssal_goto_cycle(ctx, ctx->cursor_cycle + delta);
}

void abyssal_seek_backward(abyssal_ctx_t *ctx, int64_t delta) {
    if (delta > (int64_t)ctx->cursor_cycle)
        abyssal_goto_cycle(ctx, 0);
    else
        abyssal_goto_cycle(ctx, ctx->cursor_cycle - delta);
}

void abyssal_goto_next_edge(abyssal_ctx_t *ctx, wire_id_t wire) {
    if (!ctx)
        return;
    
    /* Find the signal */
    for (uint32_t i = 0; i < ctx->waveforms.num_signals; i++) {
        waveform_signal_t *sig = &ctx->waveforms.signals[i];
        if (sig->wire != wire)
            continue;
        
        /* Find next change after cursor */
        for (uint32_t j = 0; j < sig->num_changes; j++) {
            if (sig->changes[j].cycle > ctx->cursor_cycle) {
                abyssal_goto_cycle(ctx, sig->changes[j].cycle);
                snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                        "Next edge: %s → %d at cycle %lu",
                        sig->name, sig->changes[j].value, sig->changes[j].cycle);
                return;
            }
        }
        
        snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                "No more edges on %s", sig->name);
        return;
    }
}

void abyssal_goto_prev_edge(abyssal_ctx_t *ctx, wire_id_t wire) {
    if (!ctx)
        return;
    
    for (uint32_t i = 0; i < ctx->waveforms.num_signals; i++) {
        waveform_signal_t *sig = &ctx->waveforms.signals[i];
        if (sig->wire != wire)
            continue;
        
        /* Find previous change before cursor */
        for (int j = sig->num_changes - 1; j >= 0; j--) {
            if (sig->changes[j].cycle < ctx->cursor_cycle) {
                abyssal_goto_cycle(ctx, sig->changes[j].cycle);
                snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                        "Prev edge: %s → %d at cycle %lu",
                        sig->name, sig->changes[j].value, sig->changes[j].cycle);
                return;
            }
        }
        
        snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                "No earlier edges on %s", sig->name);
        return;
    }
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * SIGNAL TRACING
 * ════════════════════════════════════════════════════════════════════════════
 */

signal_trace_t *abyssal_trace_forward(abyssal_ctx_t *ctx, wire_id_t wire, uint64_t cycle) {
    if (!ctx || !ctx->fpga)
        return NULL;
    
    signal_trace_t *trace = calloc(1, sizeof(signal_trace_t));
    if (!trace)
        return NULL;
    
    trace->source_wire = wire;
    trace->source_cycle = cycle;
    trace->source_value = abyssal_get_signal_at(ctx, wire, cycle);
    
    trace->max_steps = 256;
    trace->steps = calloc(trace->max_steps, sizeof(trace_step_t));
    if (!trace->steps) {
        free(trace);
        return NULL;
    }
    
    /* BFS through the circuit to find all affected signals */
    bool *visited = calloc(ctx->fpga->header->num_wires, sizeof(bool));
    wire_id_t *queue = malloc(ctx->fpga->header->num_wires * sizeof(wire_id_t));
    int queue_head = 0, queue_tail = 0;
    
    /* Start from source wire */
    queue[queue_tail++] = wire;
    visited[wire] = true;
    
    /* Add initial step */
    trace->steps[trace->num_steps].wire = wire;
    trace->steps[trace->num_steps].lut_idx = -1;
    trace->steps[trace->num_steps].ff_idx = -1;
    trace->steps[trace->num_steps].arrival_cycle = cycle;
    trace->steps[trace->num_steps].value = trace->source_value;
    trace->steps[trace->num_steps].depth = 0;
    trace->num_steps++;
    
    while (queue_head < queue_tail && trace->num_steps < trace->max_steps) {
        wire_id_t current = queue[queue_head++];
        uint8_t current_depth = 0;
        
        /* Find depth of current wire */
        for (uint32_t i = 0; i < trace->num_steps; i++) {
            if (trace->steps[i].wire == current) {
                current_depth = trace->steps[i].depth;
                break;
            }
        }
        
        /* Find LUTs that use this wire as input */
        for (uint32_t i = 0; i < ctx->fpga->header->num_luts; i++) {
            gpufpga_lut_t *lut = &ctx->fpga->luts[i];
            bool uses_wire = false;
            
            for (int j = 0; j < lut->num_inputs; j++) {
                if (lut->inputs[j] == current) {
                    uses_wire = true;
                    break;
                }
            }
            
            if (uses_wire && !visited[lut->output]) {
                visited[lut->output] = true;
                queue[queue_tail++] = lut->output;
                
                trace->steps[trace->num_steps].wire = lut->output;
                trace->steps[trace->num_steps].lut_idx = i;
                trace->steps[trace->num_steps].ff_idx = -1;
                trace->steps[trace->num_steps].arrival_cycle = cycle;
                trace->steps[trace->num_steps].depth = current_depth + 1;
                trace->num_steps++;
                
                if (current_depth + 1 > trace->max_depth)
                    trace->max_depth = current_depth + 1;
            }
        }
        
        /* Find FFs that use this wire as D input */
        for (uint32_t i = 0; i < ctx->fpga->header->num_ffs; i++) {
            gpufpga_ff_t *ff = &ctx->fpga->ffs[i];
            
            if (ff->d_input == current && !visited[ff->q_output]) {
                visited[ff->q_output] = true;
                /* Don't add to queue - FF breaks combinational path */
                
                trace->steps[trace->num_steps].wire = ff->q_output;
                trace->steps[trace->num_steps].lut_idx = -1;
                trace->steps[trace->num_steps].ff_idx = i;
                trace->steps[trace->num_steps].arrival_cycle = cycle + 1;
                trace->steps[trace->num_steps].depth = 0;  /* Reset after FF */
                trace->num_steps++;
                
                trace->propagation_time = 1;  /* At least 1 cycle */
            }
        }
    }
    
    free(visited);
    free(queue);
    
    return trace;
}

void abyssal_free_trace(signal_trace_t *trace) {
    if (trace) {
        free(trace->steps);
        free(trace);
    }
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * FAULT ANALYSIS
 * ════════════════════════════════════════════════════════════════════════════
 */

const char *abyssal_fault_name(fault_type_t type) {
    static const char *names[] = {
        "None",
        "Glitch",
        "Setup Violation",
        "Hold Violation",
        "Metastability",
        "Unknown State",
        "Floating Signal",
        "Contention",
        "Combinational Loop",
        "Fanout Exceeded",
        "Unconnected",
        "Stuck-at-0",
        "Stuck-at-1",
        "Race Condition",
        "Async Crossing"
    };
    
    if (type < FAULT_COUNT)
        return names[type];
    return "Unknown";
}

const char *abyssal_fault_severity_name(fault_severity_t sev) {
    static const char *names[] = { "Info", "Warning", "Error", "Critical" };
    if (sev <= SEVERITY_CRITICAL)
        return names[sev];
    return "Unknown";
}

int abyssal_analyze_faults(abyssal_ctx_t *ctx) {
    if (!ctx)
        return -1;
    
    int faults_found = 0;
    
    /* Check for stuck-at faults */
    for (uint32_t i = 0; i < ctx->waveforms.num_signals; i++) {
        waveform_signal_t *sig = &ctx->waveforms.signals[i];
        
        if (sig->num_transitions == 0 && ctx->waveforms.total_cycles > 10) {
            if (ctx->waveforms.num_faults < ctx->waveforms.max_faults) {
                fault_record_t *f = &ctx->waveforms.faults[ctx->waveforms.num_faults++];
                
                sig_value_t val = SIG_X;
                if (sig->num_changes > 0)
                    val = sig->changes[0].value;
                
                f->type = (val == SIG_1) ? FAULT_STUCK_AT_1 : FAULT_STUCK_AT_0;
                f->severity = SEVERITY_WARNING;
                f->cycle = 0;
                f->wire = sig->wire;
                snprintf(f->message, sizeof(f->message),
                        "Signal '%s' stuck at %d for %lu cycles",
                        sig->name, val, ctx->waveforms.total_cycles);
                faults_found++;
            }
        }
        
        /* Update duty cycle */
        if (sig->time_high + sig->time_low > 0) {
            sig->duty_cycle = (float)sig->time_high / (sig->time_high + sig->time_low);
        }
    }
    
    ctx->total_faults = ctx->waveforms.num_faults;
    
    return faults_found;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * UI RENDERING
 * ════════════════════════════════════════════════════════════════════════════
 */

static void render_header(abyssal_ctx_t *ctx) {
    cursor_move(1, 1);
    printf(STYLE_HEADER);
    
    /* Title */
    printf(" ABYSSAL ");
    printf(FG_CYAN_DIM "│ ");
    
    /* View mode indicator */
    printf(FG_MAGENTA_BRIGHT);
    switch (ctx->view_mode) {
        case VIEW_WAVEFORM: printf("◈ Waveform"); break;
        case VIEW_TRACE:    printf("◈ Trace"); break;
        case VIEW_FAULTS:   printf("◈ Faults"); break;
        case VIEW_STATS:    printf("◈ Stats"); break;
        default: break;
    }
    
    /* Cycle info */
    printf(FG_CYAN_DIM " │ ");
    printf(FG_WHITE "Cycle: ");
    printf(FG_CYAN_BRIGHT "%lu", ctx->cursor_cycle);
    printf(FG_WHITE_DIM "/%lu ", ctx->waveforms.end_cycle);
    
    /* Fault count */
    printf(FG_CYAN_DIM "│ ");
    if (ctx->waveforms.num_faults > 0) {
        printf(FG_RED_BRIGHT "⚠ %u faults", ctx->waveforms.num_faults);
    } else {
        printf(FG_GREEN_BRIGHT "✓ No faults");
    }
    
    /* Zoom level */
    printf(FG_CYAN_DIM " │ ");
    printf(FG_WHITE_DIM "Zoom: %dx", ctx->zoom_level);
    
    /* Fill rest of line */
    int pos = 70;  /* Approximate */
    for (int i = pos; i < ctx->term_width; i++)
        printf(" ");
    
    printf(ABYSSAL_RESET);
}

static void render_signal_char(sig_value_t val, bool is_clock) {
    switch (val) {
        case SIG_1:
            if (is_clock)
                printf(FG_MAGENTA_BRIGHT WAVE_HIGH);
            else
                printf(FG_CYAN_BRIGHT WAVE_HIGH);
            break;
        case SIG_0:
            printf(FG_BLUE_DIM WAVE_LOW);
            break;
        case SIG_X:
            printf(FG_RED_BRIGHT WAVE_UNKNOWN);
            break;
        case SIG_Z:
            printf(FG_YELLOW_BRIGHT WAVE_HIGHZ);
            break;
        default:
            printf(FG_WHITE_DIM "?");
            break;
    }
}

static void render_waveform_view(abyssal_ctx_t *ctx) {
    int start_row = 3;
    int num_visible_signals = ctx->term_height - ctx->info_pane_height - 4;
    
    /* Draw signal names and waveforms */
    for (int row = 0; row < num_visible_signals; row++) {
        uint32_t sig_idx = ctx->view_start_signal + row;
        
        cursor_move(start_row + row, 1);
        printf(BG_ABYSS);
        clear_line();
        
        if (sig_idx >= ctx->waveforms.num_signals) {
            printf(FG_WHITE_DIM "~");
            continue;
        }
        
        waveform_signal_t *sig = &ctx->waveforms.signals[sig_idx];
        
        /* Selection highlight */
        if (sig_idx == ctx->cursor_signal) {
            printf(BG_SELECTED);
        }
        
        /* Signal name */
        printf(sig_idx == ctx->cursor_signal ? FG_WHITE : FG_AQUA);
        printf(" %-*.*s ", ctx->signal_pane_width - 2, 
               ctx->signal_pane_width - 2, sig->name);
        
        /* Separator */
        printf(BG_ABYSS FG_CYAN_DIM BOX_V " ");
        
        /* Waveform */
        for (int col = 0; col < ctx->waveform_pane_width; col++) {
            uint64_t cycle = ctx->view_start_cycle + col * ctx->zoom_level;
            
            /* Cursor indicator */
            if (cycle == ctx->cursor_cycle) {
                printf(BG_SELECTED);
            } else {
                printf(BG_ABYSS);
            }
            
            if (cycle > ctx->waveforms.end_cycle) {
                printf(FG_WHITE_DIM "·");
            } else {
                sig_value_t val = abyssal_get_signal_at(ctx, sig->wire, cycle);
                
                /* Check for transition within this column (if zoomed out) */
                if (ctx->zoom_level > 1) {
                    sig_value_t next_val = abyssal_get_signal_at(ctx, sig->wire, 
                                                cycle + ctx->zoom_level - 1);
                    if (val != next_val) {
                        printf(FG_CYAN_BRIGHT WAVE_BOTH);
                        continue;
                    }
                }
                
                render_signal_char(val, sig->is_clock);
            }
        }
    }
    
    /* Time ruler */
    cursor_move(start_row + num_visible_signals, 1);
    printf(BG_TRENCH FG_WHITE_DIM);
    printf(" %-*s ", ctx->signal_pane_width - 2, "Time");
    printf(BG_ABYSS FG_CYAN_DIM BOX_V " ");
    
    for (int col = 0; col < ctx->waveform_pane_width; col++) {
        uint64_t cycle = ctx->view_start_cycle + col * ctx->zoom_level;
        
        if (col % 10 == 0) {
            char buf[16];
            snprintf(buf, sizeof(buf), "%lu", cycle);
            int len = strlen(buf);
            if (col + len < ctx->waveform_pane_width) {
                printf(FG_WHITE_DIM "%s", buf);
                col += len - 1;
            } else {
                printf(FG_CYAN_DIM "│");
            }
        } else {
            printf(FG_CYAN_DIM "·");
        }
    }
}

static void render_trace_view(abyssal_ctx_t *ctx) {
    cursor_move(3, 1);
    printf(BG_ABYSS);
    
    if (!ctx->active_trace) {
        printf(FG_WHITE_DIM "  No active trace. Press 'T' on a signal to trace forward.\n");
        return;
    }
    
    signal_trace_t *t = ctx->active_trace;
    
    printf(FG_CYAN_BRIGHT "  Signal Trace: ");
    printf(FG_WHITE "Wire %d from cycle %lu\n", t->source_wire, t->source_cycle);
    printf(FG_CYAN_DIM "  ────────────────────────────────────────\n");
    printf(FG_WHITE_DIM "  Max combinational depth: %u\n", t->max_depth);
    printf(FG_WHITE_DIM "  Propagation time: %lu cycle(s)\n\n", t->propagation_time);
    
    printf(FG_CYAN_BRIGHT "  %s Trace Path:\n\n", ICON_TRACE);
    
    for (uint32_t i = 0; i < t->num_steps && i < 20; i++) {
        trace_step_t *step = &t->steps[i];
        
        /* Indentation based on depth */
        printf("  ");
        for (uint8_t d = 0; d < step->depth; d++)
            printf("  ");
        
        /* Icon based on source */
        if (step->ff_idx != (uint32_t)-1) {
            printf(FG_MAGENTA_BRIGHT "◉ ");  /* FF */
        } else if (step->lut_idx != (uint32_t)-1) {
            printf(FG_CYAN_BRIGHT "○ ");     /* LUT */
        } else {
            printf(FG_GREEN_BRIGHT "● ");    /* Source */
        }
        
        /* Wire info */
        printf(FG_WHITE "Wire %d", step->wire);
        
        /* Source info */
        if (step->ff_idx != (uint32_t)-1) {
            printf(FG_MAGENTA_DIM " (FF %d, next cycle)", step->ff_idx);
        } else if (step->lut_idx != (uint32_t)-1) {
            printf(FG_CYAN_DIM " (via LUT %d)", step->lut_idx);
        } else {
            printf(FG_GREEN_DIM " (source)");
        }
        
        /* Value */
        printf(FG_WHITE_DIM " = ");
        if (step->value == SIG_1) printf(FG_CYAN_BRIGHT "1");
        else if (step->value == SIG_0) printf(FG_BLUE_DIM "0");
        else printf(FG_RED_BRIGHT "X");
        
        printf("\n");
    }
    
    if (t->num_steps > 20) {
        printf(FG_WHITE_DIM "\n  ... and %u more steps\n", t->num_steps - 20);
    }
}

static void render_fault_view(abyssal_ctx_t *ctx) {
    cursor_move(3, 1);
    printf(BG_ABYSS);
    
    if (ctx->waveforms.num_faults == 0) {
        printf(FG_GREEN_BRIGHT "  ✓ No faults detected\n\n");
        printf(FG_WHITE_DIM "  Circuit is operating within expected parameters.\n");
        return;
    }
    
    printf(FG_RED_BRIGHT "  ⚠ Faults Detected: %u\n\n", ctx->waveforms.num_faults);
    
    printf(FG_CYAN_DIM "  %-8s %-16s %-10s %s\n", "Cycle", "Type", "Severity", "Message");
    printf(FG_CYAN_DIM "  ────────────────────────────────────────────────────────────\n");
    
    for (uint32_t i = 0; i < ctx->waveforms.num_faults && i < 15; i++) {
        fault_record_t *f = &ctx->waveforms.faults[i];
        
        /* Severity color */
        switch (f->severity) {
            case SEVERITY_INFO:    printf(FG_WHITE_DIM); break;
            case SEVERITY_WARNING: printf(FG_YELLOW_BRIGHT); break;
            case SEVERITY_ERROR:   printf(FG_ORANGE); break;
            case SEVERITY_CRITICAL:printf(FG_RED_BRIGHT); break;
        }
        
        printf("  %-8lu %-16s %-10s %s\n",
               f->cycle,
               abyssal_fault_name(f->type),
               abyssal_fault_severity_name(f->severity),
               f->message);
    }
}

static void render_info_pane(abyssal_ctx_t *ctx) {
    int start_row = ctx->term_height - ctx->info_pane_height;
    
    /* Border */
    cursor_move(start_row, 1);
    printf(BG_TRENCH FG_CYAN_DIM);
    for (int i = 0; i < ctx->term_width; i++)
        printf(BOX_H);
    
    /* Selected signal info */
    cursor_move(start_row + 1, 1);
    printf(BG_ABYSS);
    clear_line();
    
    if (ctx->cursor_signal < ctx->waveforms.num_signals) {
        waveform_signal_t *sig = &ctx->waveforms.signals[ctx->cursor_signal];
        
        printf(FG_CYAN_BRIGHT " %s ", sig->name);
        printf(FG_WHITE_DIM "(Wire %d) ", sig->wire);
        
        sig_value_t val = abyssal_get_signal_at(ctx, sig->wire, ctx->cursor_cycle);
        printf(FG_WHITE "Current: ");
        if (val == SIG_1) printf(FG_CYAN_BRIGHT "HIGH ");
        else if (val == SIG_0) printf(FG_BLUE_DIM "LOW ");
        else printf(FG_RED_BRIGHT "UNKNOWN ");
        
        printf(FG_WHITE_DIM "│ Transitions: %u ", sig->num_transitions);
        printf(FG_WHITE_DIM "│ Glitches: ");
        if (sig->num_glitches > 0)
            printf(FG_YELLOW_BRIGHT "%u ", sig->num_glitches);
        else
            printf(FG_GREEN_DIM "0 ");
        
        printf(FG_WHITE_DIM "│ Duty: %.1f%%", sig->duty_cycle * 100);
    }
    
    /* Help hints */
    cursor_move(start_row + 2, 1);
    printf(BG_ABYSS);
    clear_line();
    printf(FG_CYAN_DIM " [←→] Scrub time  [↑↓] Select signal  ");
    printf("[n/p] Next/prev edge  [t] Trace  [Space] Step  [r] Run  [q] Quit");
}

static void render_status_bar(abyssal_ctx_t *ctx) {
    cursor_move(ctx->term_height, 1);
    printf(BG_TRENCH);
    clear_line();
    
    if (ctx->command_mode) {
        printf(FG_CYAN_BRIGHT " : ");
        printf(FG_WHITE "%s", ctx->command_buf);
        printf(FG_CYAN_BRIGHT "█");
    } else {
        printf(FG_WHITE " %s", ctx->status_msg);
    }
    
    /* Right side: simulation status */
    cursor_move(ctx->term_height, ctx->term_width - 20);
    if (ctx->running) {
        printf(FG_GREEN_BRIGHT " ▶ Running ");
    } else {
        printf(FG_YELLOW_BRIGHT " ⏸ Paused ");
    }
}

void abyssal_render(abyssal_ctx_t *ctx) {
    if (!ctx)
        return;
    
    get_terminal_size(&ctx->term_width, &ctx->term_height);
    ctx->waveform_pane_width = ctx->term_width - ctx->signal_pane_width - 3;
    
    clear_screen();
    render_header(ctx);
    
    switch (ctx->view_mode) {
        case VIEW_WAVEFORM:
            render_waveform_view(ctx);
            break;
        case VIEW_TRACE:
            render_trace_view(ctx);
            break;
        case VIEW_FAULTS:
            render_fault_view(ctx);
            break;
        default:
            render_waveform_view(ctx);
            break;
    }
    
    render_info_pane(ctx);
    render_status_bar(ctx);
    
    printf(ABYSSAL_RESET);
    fflush(stdout);
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * INPUT HANDLING
 * ════════════════════════════════════════════════════════════════════════════
 */

int abyssal_handle_key(abyssal_ctx_t *ctx, int key) {
    if (!ctx)
        return -1;
    
    if (ctx->command_mode) {
        if (key == 27) {  /* Escape */
            ctx->command_mode = false;
            ctx->command_len = 0;
            ctx->command_buf[0] = '\0';
        } else if (key == '\r' || key == '\n') {
            ctx->command_mode = false;
            abyssal_handle_command(ctx, ctx->command_buf);
            ctx->command_len = 0;
            ctx->command_buf[0] = '\0';
        } else if (key == 127 || key == 8) {  /* Backspace */
            if (ctx->command_len > 0) {
                ctx->command_buf[--ctx->command_len] = '\0';
            }
        } else if (key >= 32 && key < 127 && ctx->command_len < 250) {
            ctx->command_buf[ctx->command_len++] = key;
            ctx->command_buf[ctx->command_len] = '\0';
        }
        return 0;
    }
    
    switch (key) {
        case 'q':
        case 'Q':
            return 1;  /* Quit */
        
        case ':':
            ctx->command_mode = true;
            ctx->command_len = 0;
            ctx->command_buf[0] = '\0';
            break;
        
        /* Navigation */
        case 'h':
        case 'D':  /* Left arrow */
            abyssal_seek_backward(ctx, ctx->zoom_level);
            break;
        
        case 'l':
        case 'C':  /* Right arrow */
            abyssal_seek_forward(ctx, ctx->zoom_level);
            break;
        
        case 'k':
        case 'A':  /* Up arrow */
            if (ctx->cursor_signal > 0)
                ctx->cursor_signal--;
            break;
        
        case 'j':
        case 'B':  /* Down arrow */
            if (ctx->cursor_signal < ctx->waveforms.num_signals - 1)
                ctx->cursor_signal++;
            break;
        
        case 'H':  /* Fast left */
            abyssal_seek_backward(ctx, 10 * ctx->zoom_level);
            break;
        
        case 'L':  /* Fast right */
            abyssal_seek_forward(ctx, 10 * ctx->zoom_level);
            break;
        
        case 'g':  /* Go to start */
            abyssal_goto_cycle(ctx, ctx->waveforms.start_cycle);
            break;
        
        case 'G':  /* Go to end */
            abyssal_goto_cycle(ctx, ctx->waveforms.end_cycle);
            break;
        
        /* Edge navigation */
        case 'n':  /* Next edge */
            if (ctx->cursor_signal < ctx->waveforms.num_signals) {
                abyssal_goto_next_edge(ctx, 
                    ctx->waveforms.signals[ctx->cursor_signal].wire);
            }
            break;
        
        case 'p':  /* Previous edge */
            if (ctx->cursor_signal < ctx->waveforms.num_signals) {
                abyssal_goto_prev_edge(ctx,
                    ctx->waveforms.signals[ctx->cursor_signal].wire);
            }
            break;
        
        /* Zoom */
        case '+':
        case '=':
            if (ctx->zoom_level > 1)
                ctx->zoom_level--;
            snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                    "Zoom: %dx", ctx->zoom_level);
            break;
        
        case '-':
            if (ctx->zoom_level < 100)
                ctx->zoom_level++;
            snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                    "Zoom: %dx", ctx->zoom_level);
            break;
        
        /* Simulation */
        case ' ':  /* Step */
            abyssal_step(ctx);
            snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                    "Stepped to cycle %lu", ctx->fpga->cycles_simulated);
            break;
        
        case 'r':  /* Run */
            ctx->running = !ctx->running;
            if (ctx->running) {
                abyssal_run_to(ctx, ctx->cursor_cycle + 100);
                ctx->running = false;
            }
            break;
        
        /* Trace */
        case 't':
        case 'T':
            if (ctx->cursor_signal < ctx->waveforms.num_signals) {
                if (ctx->active_trace)
                    abyssal_free_trace(ctx->active_trace);
                ctx->active_trace = abyssal_trace_forward(ctx,
                    ctx->waveforms.signals[ctx->cursor_signal].wire,
                    ctx->cursor_cycle);
                ctx->view_mode = VIEW_TRACE;
                snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                        "Tracing signal forward from cycle %lu", ctx->cursor_cycle);
            }
            break;
        
        /* View modes */
        case '1':
            ctx->view_mode = VIEW_WAVEFORM;
            break;
        case '2':
            ctx->view_mode = VIEW_TRACE;
            break;
        case '3':
            ctx->view_mode = VIEW_FAULTS;
            abyssal_analyze_faults(ctx);
            break;
        
        /* Help */
        case '?':
            snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                    "hjkl/arrows: navigate  n/p: edges  +-: zoom  space: step  t: trace  1-3: views  q: quit");
            break;
        
        default:
            break;
    }
    
    return 0;
}

int abyssal_handle_command(abyssal_ctx_t *ctx, const char *cmd) {
    if (!ctx || !cmd)
        return -1;
    
    /* Parse command */
    if (strncmp(cmd, "goto ", 5) == 0 || strncmp(cmd, "g ", 2) == 0) {
        uint64_t cycle = atoll(cmd + (cmd[1] == ' ' ? 2 : 5));
        abyssal_goto_cycle(ctx, cycle);
        snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                "Jumped to cycle %lu", cycle);
    }
    else if (strcmp(cmd, "run") == 0 || strcmp(cmd, "r") == 0) {
        abyssal_run_to(ctx, ctx->cursor_cycle + 1000);
    }
    else if (strncmp(cmd, "run ", 4) == 0) {
        uint64_t cycles = atoll(cmd + 4);
        abyssal_run_to(ctx, ctx->cursor_cycle + cycles);
    }
    else if (strcmp(cmd, "analyze") == 0 || strcmp(cmd, "a") == 0) {
        int found = abyssal_analyze_faults(ctx);
        snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                "Analysis complete: %d faults found", found);
    }
    else if (strcmp(cmd, "q") == 0 || strcmp(cmd, "quit") == 0) {
        return 1;
    }
    else {
        snprintf(ctx->status_msg, sizeof(ctx->status_msg),
                "Unknown command: %s", cmd);
    }
    
    return 0;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * MAIN LOOP
 * ════════════════════════════════════════════════════════════════════════════
 */

int abyssal_run(abyssal_ctx_t *ctx) {
    if (!ctx)
        return -1;
    
    terminal_raw_mode();
    
    /* Initial render */
    abyssal_render(ctx);
    
    /* Main loop */
    while (1) {
        /* Read input (non-blocking) */
        char c;
        if (read(STDIN_FILENO, &c, 1) == 1) {
            /* Handle escape sequences */
            if (c == 27) {
                char seq[2];
                if (read(STDIN_FILENO, &seq[0], 1) == 1 &&
                    read(STDIN_FILENO, &seq[1], 1) == 1) {
                    if (seq[0] == '[') {
                        c = seq[1];  /* A=up, B=down, C=right, D=left */
                    }
                }
            }
            
            if (abyssal_handle_key(ctx, c) != 0) {
                break;  /* Quit requested */
            }
            
            abyssal_render(ctx);
        }
        
        /* Small delay to avoid busy-waiting */
        usleep(10000);
    }
    
    terminal_restore();
    return 0;
}
