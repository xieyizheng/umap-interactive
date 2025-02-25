import torch
import psutil
import platform
from typing import Dict

def check_device_availability():
    # Check CUDA (NVIDIA GPU)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0)
        print(f"CUDA available: {cuda_available}")
        print(f"Number of CUDA devices: {cuda_device_count}")
        print(f"CUDA device name: {cuda_device_name}")
    
    # Check MPS (Apple M1/M2)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS (Apple Silicon) available: {mps_available}")
    
    # Check current device
    if cuda_available:
        device = torch.device("cuda")
    elif mps_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    return device

def get_accelerator_stats() -> Dict:
    """Monitor accelerator (GPU/MPS) memory usage and other stats."""
    stats = {}
    
    # Check CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        
        # Memory stats in GB
        allocated = torch.cuda.memory_allocated(current_device) / 1e9
        reserved = torch.cuda.memory_reserved(current_device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(current_device) / 1e9
        
        stats['cuda'] = {
            'device_name': torch.cuda.get_device_name(current_device),
            'allocated_memory_gb': f"{allocated:.2f}",
            'reserved_memory_gb': f"{reserved:.2f}",
            'peak_memory_gb': f"{max_allocated:.2f}",
            'device_capability': torch.cuda.get_device_capability(current_device)
        }
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        vm = psutil.virtual_memory()
        stats['mps'] = {
            'device': 'Apple Silicon',
            'platform': platform.processor(),
            'total_memory_gb': f"{vm.total / 1e9:.2f}",
            'available_memory_gb': f"{vm.available / 1e9:.2f}",
            'memory_percent_used': f"{vm.percent:.1f}%"
        }
    
    return stats

def print_accelerator_stats():
    """Print formatted accelerator statistics."""
    stats = get_accelerator_stats()
    
    if 'cuda' in stats:
        print("\nCUDA (NVIDIA GPU) Stats:")
        print(f"Device: {stats['cuda']['device_name']}")
        print(f"Allocated Memory: {stats['cuda']['allocated_memory_gb']} GB")
        print(f"Reserved Memory: {stats['cuda']['reserved_memory_gb']} GB")
        print(f"Peak Memory Usage: {stats['cuda']['peak_memory_gb']} GB")
        print(f"Device Capability: {stats['cuda']['device_capability']}")
    
    if 'mps' in stats:
        print("\nMPS (Apple Silicon) Stats:")
        print(f"Device: {stats['mps']['device']}")
        print(f"Platform: {stats['mps']['platform']}")
        print(f"Total Memory: {stats['mps']['total_memory_gb']} GB")
        print(f"Available Memory: {stats['mps']['available_memory_gb']} GB")
        print(f"Memory Usage: {stats['mps']['memory_percent_used']}") 