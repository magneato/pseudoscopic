#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
nmc_bench.py - Near-Memory Computing Benchmark

Measures the PCIe efficiency gains of NMC patterns vs traditional approaches.

Usage:
    python3 nmc_bench.py [--size SIZE_MB] [--ops NUM_OPS]

Copyright (C) 2025 Neural Splines LLC
"""

import argparse
import ctypes
import os
import sys
import time
import mmap
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def banner():
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   {BOLD}NMC Benchmark - Near-Memory Computing Performance{RESET}{CYAN}              ║
║                                                                    ║
║   Measuring the asymmetric advantage                               ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════╝{RESET}
""")

def check_pseudoscopic():
    """Check if pseudoscopic is loaded and in ramdisk mode."""
    devices = list(Path("/dev").glob("psdisk*")) + list(Path("/dev").glob("pswap*"))
    
    if not devices:
        print(f"{RED}✗ Pseudoscopic device not found{RESET}")
        print(f"\n  Load the driver in ramdisk mode:")
        print(f"  {YELLOW}sudo modprobe pseudoscopic mode=ramdisk{RESET}")
        return None
    
    device = devices[0]
    print(f"{GREEN}✓ Found pseudoscopic device: {device}{RESET}")
    
    # Get size
    try:
        with open(device, 'rb') as f:
            f.seek(0, 2)  # Seek to end
            size = f.tell()
        print(f"  Capacity: {size / (1024**3):.2f} GB")
        return str(device), size
    except PermissionError:
        print(f"{RED}✗ Permission denied. Run with sudo or add user to disk group.{RESET}")
        return None

def check_cuda():
    """Check CUDA availability."""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        device = cuda.Device(0)
        print(f"{GREEN}✓ CUDA available: {device.name()}{RESET}")
        print(f"  Memory: {device.total_memory() / (1024**3):.2f} GB")
        return True
    except ImportError:
        print(f"{YELLOW}! PyCUDA not installed (optional){RESET}")
        return False
    except Exception as e:
        print(f"{YELLOW}! CUDA not available: {e}{RESET}")
        return False

def benchmark_pcie_write(device_path, size_mb):
    """Benchmark write speed to VRAM via pseudoscopic."""
    size = size_mb * 1024 * 1024
    
    print(f"\n{BOLD}PCIe Write Benchmark (CPU → VRAM){RESET}")
    print(f"  Size: {size_mb} MB")
    
    # Generate test data
    data = os.urandom(size)
    
    # Open device and mmap
    fd = os.open(device_path, os.O_RDWR)
    try:
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        
        # Warm up
        mm[0:4096] = data[0:4096]
        mm.flush()
        
        # Timed write
        start = time.perf_counter()
        mm[0:size] = data
        mm.flush()
        elapsed = time.perf_counter() - start
        
        bandwidth = size / elapsed / (1024**3)
        print(f"  {GREEN}Time: {elapsed*1000:.2f} ms{RESET}")
        print(f"  {GREEN}Bandwidth: {bandwidth:.2f} GB/s{RESET}")
        
        mm.close()
        return bandwidth
    finally:
        os.close(fd)

def benchmark_pcie_read(device_path, size_mb):
    """Benchmark read speed from VRAM via pseudoscopic."""
    size = size_mb * 1024 * 1024
    
    print(f"\n{BOLD}PCIe Read Benchmark (VRAM → CPU){RESET}")
    print(f"  Size: {size_mb} MB")
    
    fd = os.open(device_path, os.O_RDWR)
    try:
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        
        # Warm up
        _ = mm[0:4096]
        
        # Timed read
        start = time.perf_counter()
        data = mm[0:size]
        elapsed = time.perf_counter() - start
        
        bandwidth = size / elapsed / (1024**3)
        print(f"  {GREEN}Time: {elapsed*1000:.2f} ms{RESET}")
        print(f"  {GREEN}Bandwidth: {bandwidth:.2f} GB/s{RESET}")
        
        mm.close()
        return bandwidth
    finally:
        os.close(fd)

def calculate_nmc_advantage(data_size_mb, num_ops, write_bw, read_bw):
    """Calculate the PCIe efficiency advantage of NMC."""
    
    print(f"\n{BOLD}═══════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}NMC Advantage Analysis{RESET}")
    print(f"{BOLD}═══════════════════════════════════════════════════════════════════{RESET}")
    
    data_size = data_size_mb * 1024 * 1024
    
    # Traditional approach: copy to GPU and back for each operation
    trad_pcie = data_size * 2 * num_ops  # To GPU + back, for each op
    trad_time = (data_size / write_bw / 1e9 + data_size / read_bw / 1e9) * num_ops
    
    # NMC approach: one load, results only
    result_size = 1024  # Assume 1KB results per operation
    nmc_pcie = data_size + result_size * num_ops
    nmc_time = data_size / write_bw / 1e9 + (result_size * num_ops) / read_bw / 1e9
    
    print(f"\nScenario: {data_size_mb} MB data, {num_ops} operations")
    print(f"\n{CYAN}Traditional Approach:{RESET}")
    print(f"  PCIe traffic: {trad_pcie / (1024**3):.2f} GB")
    print(f"  Transfer time: {trad_time:.2f} seconds")
    
    print(f"\n{CYAN}NMC Approach:{RESET}")
    print(f"  PCIe traffic: {nmc_pcie / (1024**2):.2f} MB")
    print(f"  Transfer time: {nmc_time:.2f} seconds")
    
    pcie_reduction = trad_pcie / nmc_pcie
    time_reduction = trad_time / nmc_time
    
    print(f"\n{GREEN}{BOLD}Results:{RESET}")
    print(f"  {GREEN}PCIe traffic reduction: {pcie_reduction:.1f}x{RESET}")
    print(f"  {GREEN}Transfer time reduction: {time_reduction:.1f}x{RESET}")
    
    # Additional insight
    print(f"\n{YELLOW}Insight:{RESET}")
    print(f"  For every additional operation on the same data,")
    print(f"  traditional approach moves {data_size_mb*2} MB more.")
    print(f"  NMC moves only ~{result_size/1024:.1f} KB (the results).")
    
    return pcie_reduction, time_reduction

def main():
    parser = argparse.ArgumentParser(description='NMC Performance Benchmark')
    parser.add_argument('--size', type=int, default=256,
                        help='Test data size in MB (default: 256)')
    parser.add_argument('--ops', type=int, default=10,
                        help='Number of simulated operations (default: 10)')
    args = parser.parse_args()
    
    banner()
    
    # Check prerequisites
    print(f"{BOLD}Checking prerequisites...{RESET}\n")
    
    result = check_pseudoscopic()
    if not result:
        sys.exit(1)
    
    device_path, capacity = result
    
    if args.size * 1024 * 1024 > capacity:
        print(f"\n{RED}Requested size ({args.size} MB) exceeds device capacity{RESET}")
        args.size = int(capacity / (1024 * 1024) / 2)
        print(f"  Using {args.size} MB instead")
    
    check_cuda()
    
    # Run benchmarks
    print(f"\n{BOLD}Running benchmarks...{RESET}")
    
    write_bw = benchmark_pcie_write(device_path, args.size)
    read_bw = benchmark_pcie_read(device_path, args.size)
    
    # Calculate advantage
    calculate_nmc_advantage(args.size, args.ops, write_bw, read_bw)
    
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   The fastest data transfer is the one that doesn't happen.        ║
║                                                                    ║
║                           - Asymmetric Solutions                   ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════╝{RESET}
""")

if __name__ == '__main__':
    main()
