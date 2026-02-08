# Pseudoscopic Production Release Plan

## Overview
Production preparation for the Pseudoscopic near-memory computing system.

## Completed Tasks

### 1. Build System Fixes ✓
- Fixed Makefile path issues for kernel module
- Fixed CUDA detection (using wildcard instead of which)
- Fixed setup.sh GCC detection (stderr redirection)
- Fixed DKMS installation
- Fixed CUDA runtime library linking (-lcudart)

### 2. Kernel API Compatibility ✓
- Replaced deprecated linux/gendisk.h
- Fixed blk_mq_alloc_disk arguments
- Fixed devm_request_free_mem_region arguments
- Added #ifndef guards for PCI_REBAR constants

### 3. Code Safety Fixes ✓
- Fixed container_of misuse in pool.c
- Fixed stack overflow in migrate.c
- Added missing function prototypes
- Implemented proper get_internal_ctx() accessor

### 4. TODO Resolution ✓
- ✓ nearmem_gpu.c:375 - Track allocations for free space (implemented allocation tracking)
- ✓ nearmem_gpu.c:415 - Use P2P if available (implemented P2P copy with fallback)
- ✓ nearmem.c:363 - Proper container_of (implemented get_internal_ctx() accessor)

### 5. Thermal Management ✓
- Added gpufpga_thermal_params_t with realistic FPGA parameters
- Implemented thermal zone modeling for spatial temperature tracking
- Added power consumption calculation (dynamic + static)
- Implemented thermal throttling simulation
- Added thermal statistics reporting

### 6. VHDL/Verilog Support ✓
- Added Verilog subset parser (module, input, output, wire, reg, assign)
- Implemented gate instantiation parsing (and, or, xor, not, nand, nor)
- Added VCD waveform output for GTKWave compatibility
- Added timing analysis API
- Added port mapping for testbench integration

### 7. Build Verification ✓
- Kernel module builds successfully (pseudoscopic.ko)
- Near-memory library builds (libnearmem.a, libnearmem.so)
- All examples build and link correctly

## Pending Tasks

### 8. Documentation Cleanup
- [ ] Update README.md with thermal/HDL examples
- [ ] Add API documentation for new features
- [ ] Clean up excessive ASCII art if any

### 9. Advanced Features (Future)
- [ ] VHDL parser (currently Verilog only)
- [ ] Block RAM simulation
- [ ] DSP slice simulation
- [ ] Multiple clock domain support
- [ ] nmc.c:140 - Match by PCI bus ID from sysfs

## Build Status
- Kernel module: ✓ PASS
- Near-memory library: ✓ PASS
- Examples: ✓ PASS
- CUDA integration: ✓ PASS (with runtime library)

