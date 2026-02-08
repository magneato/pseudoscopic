/* SPDX-License-Identifier: GPL-2.0 */
/*
 * hw.h - Hardware register definitions for NVIDIA GPUs
 *
 * These definitions are derived from nouveau driver sources
 * and publicly available documentation. We only define what
 * we actually use for BAR management.
 *
 * Copyright (C) 2026 Neural Splines LLC
 */

#ifndef _PSEUDOSCOPIC_HW_H_
#define _PSEUDOSCOPIC_HW_H_

/*
 * PCIe Configuration Space
 * ------------------------
 * Standard PCI configuration registers plus NVIDIA-specific
 * capabilities for BAR management.
 */

/* PCIe Resizable BAR capability ID */
#define PCI_EXT_CAP_ID_REBAR    0x15

/* Resizable BAR capability structure offsets */
#ifndef PCI_REBAR_CAP
#define PCI_REBAR_CAP           0x04    /* Capability register */
#endif
#ifndef PCI_REBAR_CTRL
#define PCI_REBAR_CTRL          0x08    /* Control register */
#endif

/* Resizable BAR control register bits */
#define PCI_REBAR_CTRL_BAR_IDX_MASK     0x00000007
#define PCI_REBAR_CTRL_BAR_SIZE_MASK    0x00001F00
#define PCI_REBAR_CTRL_BAR_SIZE_SHIFT   8

/* BAR size encoding (2^(n+20) bytes) */
#define PCI_REBAR_SIZE_1MB      0
#define PCI_REBAR_SIZE_2MB      1
#define PCI_REBAR_SIZE_4MB      2
#define PCI_REBAR_SIZE_8MB      3
#define PCI_REBAR_SIZE_16MB     4
#define PCI_REBAR_SIZE_32MB     5
#define PCI_REBAR_SIZE_64MB     6
#define PCI_REBAR_SIZE_128MB    7
#define PCI_REBAR_SIZE_256MB    8
#define PCI_REBAR_SIZE_512MB    9
#define PCI_REBAR_SIZE_1GB      10
#define PCI_REBAR_SIZE_2GB      11
#define PCI_REBAR_SIZE_4GB      12
#define PCI_REBAR_SIZE_8GB      13
#define PCI_REBAR_SIZE_16GB     14
#define PCI_REBAR_SIZE_32GB     15
#define PCI_REBAR_SIZE_64GB     16

/*
 * BAR0 MMIO Registers
 * -------------------
 * Selected registers we may need for BAR management
 * and device identification.
 */

/* Boot/identification registers */
#define NV_PMC_BOOT_0           0x00000000
#define NV_PMC_BOOT_1           0x00000004

/* Boot register fields */
#define NV_PMC_BOOT_0_MINOR_REVISION_MASK   0x0000000F
#define NV_PMC_BOOT_0_MAJOR_REVISION_MASK   0x000000F0
#define NV_PMC_BOOT_0_MAJOR_REVISION_SHIFT  4
#define NV_PMC_BOOT_0_IMPL_MASK             0x00FF0000
#define NV_PMC_BOOT_0_IMPL_SHIFT            16
#define NV_PMC_BOOT_0_ARCHITECTURE_MASK     0xFF000000
#define NV_PMC_BOOT_0_ARCHITECTURE_SHIFT    24

/* Architecture codes */
#define NV_PMC_BOOT_0_ARCH_GK100    0xE0    /* Kepler */
#define NV_PMC_BOOT_0_ARCH_GK110    0xF0    /* Kepler */
#define NV_PMC_BOOT_0_ARCH_GM100    0x110   /* Maxwell 1st gen */
#define NV_PMC_BOOT_0_ARCH_GM200    0x120   /* Maxwell 2nd gen */
#define NV_PMC_BOOT_0_ARCH_GP100    0x130   /* Pascal (P100) */
#define NV_PMC_BOOT_0_ARCH_GP102    0x132   /* Pascal (P40) */
#define NV_PMC_BOOT_0_ARCH_GV100    0x140   /* Volta (V100) */
#define NV_PMC_BOOT_0_ARCH_TU100    0x160   /* Turing */
#define NV_PMC_BOOT_0_ARCH_GA100    0x170   /* Ampere */

/*
 * FB (Framebuffer/Memory) Registers
 * ---------------------------------
 * For querying memory configuration.
 */

#define NV_PFB_CFG0             0x00100200
#define NV_PFB_CFG1             0x00100204

/*
 * PBUS Registers
 * --------------
 * Bus interface configuration.
 */

#define NV_PBUS_PCI_NV_1        0x00001804
#define NV_PBUS_PCI_NV_19       0x0000184C

/* BAR control bits */
#define NV_PBUS_PCI_NV_19_BAR1_SIZE_MASK    0x0000001F
#define NV_PBUS_PCI_NV_19_BAR1_SIZE_256MB   0x0F
#define NV_PBUS_PCI_NV_19_BAR1_SIZE_512MB   0x10
#define NV_PBUS_PCI_NV_19_BAR1_SIZE_1GB     0x11
#define NV_PBUS_PCI_NV_19_BAR1_SIZE_2GB     0x12
#define NV_PBUS_PCI_NV_19_BAR1_SIZE_4GB     0x13
#define NV_PBUS_PCI_NV_19_BAR1_SIZE_8GB     0x14
#define NV_PBUS_PCI_NV_19_BAR1_SIZE_16GB    0x15
#define NV_PBUS_PCI_NV_19_BAR1_SIZE_32GB    0x16
#define NV_PBUS_PCI_NV_19_BAR1_SIZE_64GB    0x17

/*
 * GPU Architecture Detection
 * --------------------------
 * Helper to identify GPU generation from boot register.
 */

static inline unsigned int nv_arch_from_boot0(u32 boot0)
{
    return (boot0 & NV_PMC_BOOT_0_ARCHITECTURE_MASK) >> 
           NV_PMC_BOOT_0_ARCHITECTURE_SHIFT;
}

static inline bool nv_is_pascal_or_newer(u32 boot0)
{
    return nv_arch_from_boot0(boot0) >= NV_PMC_BOOT_0_ARCH_GP100;
}

/*
 * BAR Size Helpers
 * ----------------
 * Convert between PCIe resizable BAR encoding and bytes.
 */

static inline resource_size_t rebar_size_to_bytes(unsigned int size_code)
{
    /* Size is 2^(n+20) bytes */
    return 1ULL << (size_code + 20);
}

static inline unsigned int bytes_to_rebar_size(resource_size_t bytes)
{
    unsigned int code = 0;
    resource_size_t size = 1ULL << 20;  /* 1 MB minimum */
    
    while (size < bytes && code < 16) {
        size <<= 1;
        code++;
    }
    return code;
}

#endif /* _PSEUDOSCOPIC_HW_H_ */
