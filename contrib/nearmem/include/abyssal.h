/*
 * abyssal.h - Deep-Sea Bioluminescent Circuit Debugger for gpuFPGA
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *    "In the deepest ocean, where no light reaches, life creates its own."
 *
 *    ABYSSAL is a professional-grade FPGA debugger that lets you:
 *      • Trace individual signal pulses through combinational logic
 *      • Scrub through simulation time like a video editor
 *      • Detect circuit failures (glitches, metastability, timing)
 *      • Visualize signal propagation in real-time
 *
 *    The interface is inspired by the bioluminescent creatures of the
 *    deep sea: soft glowing cyans, pulsing magentas, and the occasional
 *    bright flash of warning when something goes wrong.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#ifndef _ABYSSAL_H_
#define _ABYSSAL_H_

#include <stdint.h>
#include <stdbool.h>
#include "gpufpga.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ════════════════════════════════════════════════════════════════════════════
 * COLOR PALETTE - Deep Sea Bioluminescence
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The deep ocean is not black—it pulses with life.
 * Creatures communicate through light: cyan, blue, green, magenta.
 * Our debugger speaks the same language.
 */

/* ANSI escape sequences for 256-color and true-color terminals */
#define ABYSSAL_ESC         "\033["
#define ABYSSAL_RESET       "\033[0m"
#define ABYSSAL_BOLD        "\033[1m"
#define ABYSSAL_DIM         "\033[2m"
#define ABYSSAL_BLINK       "\033[5m"
#define ABYSSAL_REVERSE     "\033[7m"

/* Background colors - The Abyss */
#define BG_ABYSS            "\033[48;2;8;12;21m"      /* Deep ocean black */
#define BG_DEEP             "\033[48;2;12;18;32m"     /* Slightly lighter */
#define BG_TRENCH           "\033[48;2;16;24;42m"     /* Panel background */
#define BG_SELECTED         "\033[48;2;0;60;80m"      /* Selected item */
#define BG_ERROR            "\033[48;2;60;0;0m"       /* Error highlight */

/* Foreground colors - Bioluminescence */
#define FG_CYAN_BRIGHT      "\033[38;2;0;255;255m"    /* Signal HIGH */
#define FG_CYAN_DIM         "\033[38;2;0;128;160m"    /* Signal transition */
#define FG_BLUE_BRIGHT      "\033[38;2;0;160;255m"    /* Clock signals */
#define FG_BLUE_DIM         "\033[38;2;0;80;128m"     /* Signal LOW */
#define FG_GREEN_BRIGHT     "\033[38;2;0;255;136m"    /* Valid/OK */
#define FG_GREEN_DIM        "\033[38;2;0;128;68m"     /* Stable */
#define FG_MAGENTA_BRIGHT   "\033[38;2;255;0;255m"    /* Active/pulse */
#define FG_MAGENTA_DIM      "\033[38;2;128;0;128m"    /* Inactive */
#define FG_YELLOW_BRIGHT    "\033[38;2;255;200;0m"    /* Warning */
#define FG_ORANGE           "\033[38;2;255;128;0m"    /* Caution */
#define FG_RED_BRIGHT       "\033[38;2;255;60;60m"    /* Error/fault */
#define FG_WHITE            "\033[38;2;200;220;240m"  /* Text */
#define FG_WHITE_DIM        "\033[38;2;100;120;140m"  /* Dim text */
#define FG_AQUA             "\033[38;2;64;224;208m"   /* Highlights */

/* Combined styles for common elements */
#define STYLE_HEADER        BG_TRENCH FG_CYAN_BRIGHT ABYSSAL_BOLD
#define STYLE_SIGNAL_HI     BG_ABYSS FG_CYAN_BRIGHT
#define STYLE_SIGNAL_LO     BG_ABYSS FG_BLUE_DIM
#define STYLE_SIGNAL_X      BG_ABYSS FG_RED_BRIGHT
#define STYLE_SIGNAL_Z      BG_ABYSS FG_YELLOW_BRIGHT
#define STYLE_CLOCK         BG_ABYSS FG_MAGENTA_BRIGHT
#define STYLE_SELECTED      BG_SELECTED FG_WHITE ABYSSAL_BOLD
#define STYLE_ERROR         BG_ERROR FG_WHITE ABYSSAL_BOLD
#define STYLE_WARNING       BG_ABYSS FG_YELLOW_BRIGHT
#define STYLE_OK            BG_ABYSS FG_GREEN_BRIGHT
#define STYLE_DIM           BG_ABYSS FG_WHITE_DIM
#define STYLE_NORMAL        BG_ABYSS FG_WHITE

/* Box drawing characters (Unicode) */
#define BOX_H               "─"
#define BOX_V               "│"
#define BOX_TL              "┌"
#define BOX_TR              "┐"
#define BOX_BL              "└"
#define BOX_BR              "┘"
#define BOX_T               "┬"
#define BOX_B               "┴"
#define BOX_L               "├"
#define BOX_R               "┤"
#define BOX_X               "┼"
#define BOX_HH              "═"
#define BOX_VV              "║"
#define BOX_TL2             "╔"
#define BOX_TR2             "╗"
#define BOX_BL2             "╚"
#define BOX_BR2             "╝"

/* Waveform characters */
#define WAVE_HIGH           "▀"
#define WAVE_LOW            "▄"
#define WAVE_RISE           "╱"
#define WAVE_FALL           "╲"
#define WAVE_BOTH           "█"
#define WAVE_UNKNOWN        "░"
#define WAVE_HIGHZ          "┄"
#define WAVE_EDGE           "│"
#define WAVE_PULSE          "▌"
#define WAVE_GLITCH         "⚡"

/* Status indicators */
#define ICON_OK             "●"
#define ICON_WARN           "◆"
#define ICON_ERROR          "✖"
#define ICON_TRACE          "→"
#define ICON_BREAKPOINT     "◉"
#define ICON_CURSOR         "▶"
#define ICON_PULSE          "○"

/*
 * ════════════════════════════════════════════════════════════════════════════
 * SIGNAL VALUE TYPES
 * ════════════════════════════════════════════════════════════════════════════
 */

typedef enum {
    SIG_0 = 0,          /* Logic low */
    SIG_1 = 1,          /* Logic high */
    SIG_X = 2,          /* Unknown/undefined */
    SIG_Z = 3,          /* High impedance */
    SIG_W = 4,          /* Weak unknown */
    SIG_L = 5,          /* Weak low */
    SIG_H = 6,          /* Weak high */
    SIG_DASH = 7        /* Don't care */
} sig_value_t;

/*
 * ════════════════════════════════════════════════════════════════════════════
 * FAULT TYPES - What can go wrong in a circuit
 * ════════════════════════════════════════════════════════════════════════════
 */

typedef enum {
    FAULT_NONE = 0,
    
    /* Timing faults */
    FAULT_GLITCH,           /* Signal changed and returned within same cycle */
    FAULT_SETUP_VIOLATION,  /* Data changed too close to clock edge */
    FAULT_HOLD_VIOLATION,   /* Data changed too soon after clock edge */
    FAULT_METASTABILITY,    /* FF input changed during forbidden window */
    
    /* Logic faults */
    FAULT_UNKNOWN_STATE,    /* Signal is X (undefined) */
    FAULT_FLOATING,         /* Signal is Z (undriven) */
    FAULT_CONTENTION,       /* Multiple drivers fighting */
    FAULT_COMB_LOOP,        /* Combinational feedback loop */
    
    /* Structural faults */
    FAULT_FANOUT_EXCEED,    /* Too many loads on a signal */
    FAULT_UNCONNECTED,      /* Signal has no driver or no load */
    FAULT_STUCK_AT_0,       /* Signal never goes high */
    FAULT_STUCK_AT_1,       /* Signal never goes low */
    
    /* Behavioral faults */
    FAULT_RACE_CONDITION,   /* Result depends on signal arrival order */
    FAULT_ASYNC_CROSSING,   /* Signal crosses clock domains unsafely */
    
    FAULT_COUNT
} fault_type_t;

/* Fault severity */
typedef enum {
    SEVERITY_INFO = 0,
    SEVERITY_WARNING,
    SEVERITY_ERROR,
    SEVERITY_CRITICAL
} fault_severity_t;

/* Fault record */
typedef struct {
    fault_type_t    type;
    fault_severity_t severity;
    uint64_t        cycle;          /* When it occurred */
    wire_id_t       wire;           /* Which signal */
    wire_id_t       related_wire;   /* Related signal (for timing) */
    char            message[128];   /* Human-readable description */
} fault_record_t;

/*
 * ════════════════════════════════════════════════════════════════════════════
 * SIGNAL TRACE - Following a pulse through logic
 * ════════════════════════════════════════════════════════════════════════════
 */

/* A single step in a signal trace */
typedef struct {
    wire_id_t       wire;           /* Current wire */
    uint32_t        lut_idx;        /* LUT that produced this (or -1) */
    uint32_t        ff_idx;         /* FF that produced this (or -1) */
    uint64_t        arrival_cycle;  /* When the signal arrived here */
    sig_value_t     value;          /* Signal value at this point */
    uint8_t         depth;          /* Combinational depth from source */
} trace_step_t;

/* Complete trace of a signal pulse */
typedef struct {
    wire_id_t       source_wire;    /* Where the pulse originated */
    uint64_t        source_cycle;   /* When it started */
    sig_value_t     source_value;   /* The value being traced (0→1 or 1→0) */
    
    trace_step_t   *steps;          /* Array of trace steps */
    uint32_t        num_steps;
    uint32_t        max_steps;
    
    uint32_t        max_depth;      /* Deepest combinational path */
    uint64_t        propagation_time; /* Cycles to fully propagate */
} signal_trace_t;

/*
 * ════════════════════════════════════════════════════════════════════════════
 * WAVEFORM DATABASE
 * ════════════════════════════════════════════════════════════════════════════
 */

/* Single signal's waveform data */
typedef struct {
    wire_id_t       wire;
    char            name[64];       /* Signal name for display */
    uint8_t         is_clock;       /* Render differently if clock */
    uint8_t         is_bus;         /* Part of a bus */
    uint16_t        bus_width;      /* Width if bus */
    
    /* Value history (compressed: store only changes) */
    struct {
        uint64_t    cycle;
        sig_value_t value;
    } *changes;
    uint32_t        num_changes;
    uint32_t        max_changes;
    
    /* Analysis results */
    uint32_t        num_transitions;
    uint32_t        num_glitches;
    uint64_t        time_high;      /* Cycles spent high */
    uint64_t        time_low;       /* Cycles spent low */
    float           duty_cycle;
} waveform_signal_t;

/* Complete waveform database */
typedef struct {
    waveform_signal_t *signals;
    uint32_t        num_signals;
    uint32_t        max_signals;
    
    uint64_t        start_cycle;
    uint64_t        end_cycle;
    uint64_t        total_cycles;
    
    /* Faults detected */
    fault_record_t *faults;
    uint32_t        num_faults;
    uint32_t        max_faults;
} waveform_db_t;

/*
 * ════════════════════════════════════════════════════════════════════════════
 * DEBUGGER STATE
 * ════════════════════════════════════════════════════════════════════════════
 */

/* View modes */
typedef enum {
    VIEW_WAVEFORM,          /* Traditional waveform view */
    VIEW_SCHEMATIC,         /* ASCII schematic (simplified) */
    VIEW_TRACE,             /* Signal trace view */
    VIEW_FAULTS,            /* Fault list view */
    VIEW_STATS              /* Statistics view */
} view_mode_t;

/* Breakpoint */
typedef struct {
    wire_id_t       wire;
    sig_value_t     trigger_value;  /* Trigger when signal equals this */
    uint64_t        trigger_cycle;  /* Or trigger at specific cycle */
    bool            enabled;
    bool            hit;            /* Has been triggered */
    char            condition[64];  /* Optional condition expression */
} breakpoint_t;

/* Main debugger context */
typedef struct {
    /* Circuit being debugged */
    gpufpga_ctx_t  *fpga;
    
    /* Waveform data */
    waveform_db_t   waveforms;
    
    /* Current trace (if any) */
    signal_trace_t *active_trace;
    
    /* Breakpoints */
    breakpoint_t   *breakpoints;
    uint32_t        num_breakpoints;
    uint32_t        max_breakpoints;
    
    /* UI state */
    view_mode_t     view_mode;
    uint64_t        cursor_cycle;       /* Current time position */
    uint32_t        cursor_signal;      /* Selected signal index */
    uint64_t        view_start_cycle;   /* Left edge of waveform view */
    uint64_t        view_end_cycle;     /* Right edge of waveform view */
    uint32_t        view_start_signal;  /* Top signal in view */
    int             zoom_level;         /* Cycles per character */
    
    /* Terminal dimensions */
    int             term_width;
    int             term_height;
    
    /* Pane layout */
    int             signal_pane_width;  /* Width of signal name pane */
    int             waveform_pane_width;/* Width of waveform area */
    int             info_pane_height;   /* Height of bottom info pane */
    
    /* Status */
    bool            running;            /* Simulation running */
    bool            modified;           /* Unsaved changes */
    char            status_msg[256];    /* Status bar message */
    char            command_buf[256];   /* Command input buffer */
    int             command_len;
    bool            command_mode;       /* In command input mode */
    
    /* Statistics */
    uint64_t        total_faults;
    uint64_t        total_transitions;
    double          simulation_time_ms;
} abyssal_ctx_t;

/*
 * ════════════════════════════════════════════════════════════════════════════
 * API FUNCTIONS
 * ════════════════════════════════════════════════════════════════════════════
 */

/*
 * Initialization and cleanup
 */
int abyssal_init(abyssal_ctx_t *ctx, gpufpga_ctx_t *fpga);
void abyssal_shutdown(abyssal_ctx_t *ctx);

/*
 * Waveform management
 */
int abyssal_add_signal(abyssal_ctx_t *ctx, wire_id_t wire, const char *name);
int abyssal_add_bus(abyssal_ctx_t *ctx, wire_id_t *wires, int width, const char *name);
int abyssal_capture_cycle(abyssal_ctx_t *ctx);
void abyssal_clear_waveforms(abyssal_ctx_t *ctx);

/*
 * Simulation control
 */
int abyssal_step(abyssal_ctx_t *ctx);
int abyssal_run_to(abyssal_ctx_t *ctx, uint64_t cycle);
int abyssal_run_until_fault(abyssal_ctx_t *ctx);
int abyssal_run_until_breakpoint(abyssal_ctx_t *ctx);

/*
 * Timeline navigation (scrubbing)
 */
void abyssal_goto_cycle(abyssal_ctx_t *ctx, uint64_t cycle);
void abyssal_seek_forward(abyssal_ctx_t *ctx, int64_t delta);
void abyssal_seek_backward(abyssal_ctx_t *ctx, int64_t delta);
void abyssal_goto_next_edge(abyssal_ctx_t *ctx, wire_id_t wire);
void abyssal_goto_prev_edge(abyssal_ctx_t *ctx, wire_id_t wire);
void abyssal_goto_fault(abyssal_ctx_t *ctx, uint32_t fault_idx);

/*
 * Signal tracing
 */
signal_trace_t *abyssal_trace_forward(abyssal_ctx_t *ctx, wire_id_t wire, uint64_t cycle);
signal_trace_t *abyssal_trace_backward(abyssal_ctx_t *ctx, wire_id_t wire, uint64_t cycle);
void abyssal_free_trace(signal_trace_t *trace);

/*
 * Fault analysis
 */
int abyssal_analyze_faults(abyssal_ctx_t *ctx);
int abyssal_check_timing(abyssal_ctx_t *ctx, wire_id_t data, wire_id_t clock);
const fault_record_t *abyssal_get_fault(abyssal_ctx_t *ctx, uint32_t idx);

/*
 * Breakpoints
 */
int abyssal_add_breakpoint(abyssal_ctx_t *ctx, wire_id_t wire, sig_value_t value);
int abyssal_add_cycle_breakpoint(abyssal_ctx_t *ctx, uint64_t cycle);
void abyssal_remove_breakpoint(abyssal_ctx_t *ctx, uint32_t idx);
void abyssal_enable_breakpoint(abyssal_ctx_t *ctx, uint32_t idx, bool enable);

/*
 * UI rendering
 */
void abyssal_render(abyssal_ctx_t *ctx);
void abyssal_render_waveform_view(abyssal_ctx_t *ctx);
void abyssal_render_trace_view(abyssal_ctx_t *ctx);
void abyssal_render_fault_view(abyssal_ctx_t *ctx);
void abyssal_render_stats_view(abyssal_ctx_t *ctx);

/*
 * Input handling
 */
int abyssal_handle_key(abyssal_ctx_t *ctx, int key);
int abyssal_handle_command(abyssal_ctx_t *ctx, const char *cmd);

/*
 * Main loop
 */
int abyssal_run(abyssal_ctx_t *ctx);

/*
 * Utility functions
 */
const char *abyssal_fault_name(fault_type_t type);
const char *abyssal_fault_severity_name(fault_severity_t sev);
sig_value_t abyssal_get_signal_at(abyssal_ctx_t *ctx, wire_id_t wire, uint64_t cycle);

#ifdef __cplusplus
}
#endif

#endif /* _ABYSSAL_H_ */
