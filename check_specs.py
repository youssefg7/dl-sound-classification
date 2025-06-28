#!/usr/bin/env python3
"""Check system specifications."""

import torch
import platform
import psutil
import os

print("🖥️  System Specifications")
print("=" * 40)

# System info
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")

# CPU info
print(f"\n💻 CPU: {platform.processor()}")
print(f"Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")

# Memory info
memory = psutil.virtual_memory()
print(f"\n💾 Memory: {memory.total / (1024**3):.1f} GB total")
print(f"Available: {memory.available / (1024**3):.1f} GB")

# GPU info
print(f"\n🎮 GPU:")
if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        print(f"Compute: {props.major}.{props.minor}")
else:
    print("No CUDA GPU available")

# SLURM info
job_id = os.getenv('SLURM_JOB_ID')
if job_id:
    print(f"\n🔧 SLURM Job: {job_id}")
    print(f"Partition: {os.getenv('SLURM_JOB_PARTITION', 'Unknown')}")
    print(f"Node: {os.getenv('SLURM_JOB_NODELIST', 'Unknown')}") 