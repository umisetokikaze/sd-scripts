import functools
import gc
from typing import Optional, Union

import torch


try:
    # intel gpu support for pytorch older than 2.5
    # ipex is not needed after pytorch 2.5
    import intel_extension_for_pytorch as ipex  # noqa
except Exception:
    pass


try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_MPS = False

try:
    HAS_XPU = torch.xpu.is_available()
except Exception:
    HAS_XPU = False


def clean_memory():
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    if HAS_XPU:
        torch.xpu.empty_cache()
    if HAS_MPS:
        torch.mps.empty_cache()


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@functools.lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    r"""
    Do not call this function from training scripts. Use accelerator.device instead.
    """
    if HAS_CUDA:
        device = torch.device("cuda")
    elif HAS_XPU:
        device = torch.device("xpu")
    elif HAS_MPS:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"get_preferred_device() -> {device}")
    return device



def _normalize_cuda_arch(arch) -> Optional[str]:
    if isinstance(arch, str):
        return arch if arch.startswith("sm_") else None
    if isinstance(arch, (tuple, list)) and len(arch) >= 2:
        return f"sm_{int(arch[0])}{int(arch[1])}"
    return None


def validate_cuda_device_compatibility(device: Optional[Union[str, torch.device]] = None):
    if not HAS_CUDA:
        return

    if device is None:
        device = torch.device("cuda")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda":
        return

    get_arch_list = getattr(torch.cuda, "get_arch_list", None)
    if get_arch_list is None:
        return

    try:
        supported_arches = sorted(
            {arch_name for arch_name in (_normalize_cuda_arch(arch) for arch in get_arch_list()) if arch_name is not None}
        )
        device_arch = _normalize_cuda_arch(torch.cuda.get_device_capability(device))
        device_name = torch.cuda.get_device_name(device)
    except Exception:
        return

    if supported_arches and device_arch is not None and device_arch not in supported_arches:
        cuda_version = getattr(torch.version, "cuda", None)
        cuda_suffix = f" with CUDA {cuda_version}" if cuda_version else ""
        supported = ", ".join(supported_arches)
        raise RuntimeError(
            f"CUDA device '{device_name}' reports {device_arch}, but this PyTorch build{cuda_suffix} only supports {supported}. "
            + "Install a PyTorch build that includes kernels for this GPU from https://pytorch.org/get-started/locally/ or build PyTorch from source."
        )

def init_ipex():
    """
    Apply IPEX to CUDA hijacks using `library.ipex.ipex_init`.

    This function should run right after importing torch and before doing anything else.

    If xpu is not available, this function does nothing.
    """
    try:
        if HAS_XPU:
            from library.ipex import ipex_init

            is_initialized, error_message = ipex_init()
            if not is_initialized:
                print("failed to initialize ipex:", error_message)
        else:
            return
    except Exception as e:
        print("failed to initialize ipex:", e)
