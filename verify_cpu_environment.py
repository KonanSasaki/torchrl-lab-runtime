"""Fail the Binder image build if Python, the CPU wheel, or imports are wrong."""
import importlib.metadata
import sys

import ipykernel
import jupyter_server
import torch
import torchrl
import tensordict
from tensordict import TensorDict
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

assert sys.version_info[:2] == (3, 12), sys.version
expected = {"torch": "2.14.0+cpu", "torchrl": "0.14.0", "tensordict": "0.14.2"}
actual = {name: importlib.metadata.version(name) for name in expected}
assert actual == expected, actual
assert torch.version.cuda is None, "Expected a CPU build without a CUDA runtime"
assert not torch.cuda.is_available()
gpu_packages = sorted(
    d.metadata["Name"]
    for d in importlib.metadata.distributions()
    if d.metadata["Name"].lower().replace("_", "-").startswith("nvidia-")
    or d.metadata["Name"].lower() in {"triton", "pytorch-triton"}
)
assert not gpu_packages, f"Unexpected GPU dependency packages: {gpu_packages}"

td = TensorDict({"observation": torch.ones(2, 3)}, batch_size=[2])
assert td[1]["observation"].shape == torch.Size([3])
print("Verified Python 3.12, CPU-only TorchRL environment:", actual)
