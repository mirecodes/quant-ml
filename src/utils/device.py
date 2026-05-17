# src/utils/device.py
import torch
import platform
import psutil

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_optimal_batch_size(base_batch_size: int = 64) -> int:
    """M1 Pro 메모리에 따라 batch_size 자동 조정."""
    mem_gb = psutil.virtual_memory().total / 1e9
    
    if mem_gb >= 32:
        return base_batch_size * 2      # M1 Pro 32GB → 128
    elif mem_gb >= 16:
        return base_batch_size           # M1 Pro 16GB → 64
    else:
        return base_batch_size // 2      # 그 외 → 32

def report_environment():
    """학습 시작 전 환경 정보 출력."""
    print(f"Platform: {platform.machine()} / {platform.system()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical")
    print(f"RAM total: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"Device: {get_device()}")
    print(f"Recommended batch_size: {get_optimal_batch_size()}")

def to_device(tensor_or_model, device=None):
    if device is None:
        device = get_device()
    return tensor_or_model.to(device)
