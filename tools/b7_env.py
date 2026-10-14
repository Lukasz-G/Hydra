"""Record what box batch 7 ran on, for the cost table's caption.

A cost comparison is meaningless without the hardware it was measured on,
and batch5's hardware is only recoverable from the vast.ai contract id.
"""
import os
import platform

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("gpu_memory_GiB %.1f" % (props.total_memory / 1024 ** 3))
print("cpu_count", os.cpu_count())
print("platform", platform.platform())
