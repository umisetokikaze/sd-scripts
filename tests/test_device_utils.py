import pytest
import torch

from library import device_utils


def test_validate_cuda_device_compatibility_raises_for_unsupported_arch(monkeypatch):
    monkeypatch.setattr(device_utils, "HAS_CUDA", True)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Blackwell Test GPU")
    monkeypatch.setattr(torch.version, "cuda", "12.4", raising=False)

    with pytest.raises(RuntimeError, match="sm_120"):
        device_utils.validate_cuda_device_compatibility("cuda")


def test_validate_cuda_device_compatibility_allows_supported_arch(monkeypatch):
    monkeypatch.setattr(device_utils, "HAS_CUDA", True)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Hopper Test GPU")

    device_utils.validate_cuda_device_compatibility("cuda")
