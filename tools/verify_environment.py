#!/usr/bin/env python3
"""
Simple environment check script.
"""

# TODO The most common cause of failure is not installing PyTorch
# The README.md instructs how to do it, but people might miss it
# TODO: 1. If PyTorch or Lightning not there, direct people to that part of README
#       2. Add a unit test that does a basic smoke test of PyTorch and Lightning

import sys
import importlib

missing_dep = False

def check(pkg_name, import_name=None, extra=None):
    name = import_name or pkg_name
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", None)
        print(f"{pkg_name}: OK", end="")
        if ver:
            print(f" (version {ver})")
        else:
            print()
        if extra:
            extra(mod)
    except Exception as e:
        print(f"{pkg_name}: NOT FOUND ({e})")
        global missing_dep
        missing_dep = True

def torch_extra(torch):
    try:
        print(f"  torch version: {torch.__version__}")
        print(f"  built with CUDA: {torch.version.cuda}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"  torch extra info failed: {e}")

def lightning_extra(lightning):
    try:
        print(f"  lightning version: {getattr(lightning, '__version__', 'unknown')}")
    except Exception:
        pass

print("Python version:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
print()

check("PyTorch", "torch", torch_extra)
check("Lightning", "lightning", lightning_extra)
check("datasets")
check("heapdict")
check("tqdm")
check("rich")
check("msgpack")
print()

if missing_dep:
    print("Some dependencies missing. See README.md for installation instructions.")
    print("Alternatively, the code may work (with less functionality) even without the dep.\n")
    exit(1)
else:
    print("All depencies installed. You're good to go. Run the unit tests via `pytest`.\n")
