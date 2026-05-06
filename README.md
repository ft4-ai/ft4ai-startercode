# From Tensors to Turing Tests

![From Tensors to Turing Tests](doc/img/ft4ai_logo.png)

**From Tensors to Turing Tests** teaches generative AI and LLMs, theory and practice, using a series of hands on guided 
activities called labs. This repo contains the tools and starter code you will use. After you've cloned 
this repo, you should [install](#install) and [verify](#verify) your environment as described below, and 
then [start the first lab](#start-the-labs).

All the labs can be done on a laptop with a GPU or a simple cloud GPU, or, if with some 
adjustment, on a CPU. Linux, macOS, Windows (WSL recommended) are all supported.

Follow the installations instructions for your platform below.

# Install

Installation is via `pip` or `uv`. The specific command depends on your platform:

## You have a standard (i.e. CUDA) GPU

If you have a standard NVIDIA GPU, and have already installed the CUDA drivers, installation is simple:

First, verify your CUDA install:
```bash
$ nvidia-smi
```
Versions 12, 13, or newer will work. If you haven't installed the CUDA drivers, see:
* [WSL instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html); also see [Microsoft's guide](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
    * On WSL, do *not* install a Linux NVIDIA driver inside WSL; use the Windows driver above
* [Ubuntu (native) instructions](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers)
* [General NVIDIA instructions](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/)

Then install requirements via `pip`:
```bash
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $ python -m pip install -r requirements-cu128.txt
```

Or install via `uv` (pip-compatible but faster), which automatically creates the `.venv` and runs inside it:
```bash
$ pipx install uv # If you haven't previously installed uv
# No need to manually create a .venv
$ uv sync --extra torch-cu128
```

## You don't have any GPU

You can run this course without any GPU. Each model has adjustable size and datasets, 
and if you keep them small, they'll run fine on a CPU. Output may not be as impressive, but so what?

Via `pip`:
```bash
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $ python -m pip install -r requirements-cpu.txt
```

Or via `uv`:
```bash
$ pipx install uv # If you haven't previously installed uv
$ uv sync --extra torch-cpu
```

## You're running on a cloud GPU

Cloud GPUs, such as [RunPod](https://www.runpod.io), usually include a system-wide PyTorch you can use.

Via `pip`:
```bash
$ python -m venv --system-site-packages .venv # Allow the venv to use the system-wide packages
$ source .venv/bin/activate
(.venv) $ python -m pip install -r requirements-external.txt
```

Or via `uv`:
```bash
$ pip install --no-cache-dir uv            # If you haven't previously installed uv
$ uv venv --system-site-packages           # Tell uv to use the system-wide packages
$ uv sync --extra torch-external --inexact # and not to install PyTorch separately
```

You can use the provided `tools/container-setup.sh` to automatically setup the Cloud GPU and pull your code.

## You have a AMD ROCm, Intel GPU, Apple Silicon, or other non-standard GPU

Manually install the appropriate PyTorch. See https://pytorch.org/get-started/locally/ .

Via `pip`:
```bash
$ python -m venv .venv
$ source .venv/bin/activate
# Install PyTorch manually. Afterwards:
(.venv) $ python -m pip install -r requirements-external.txt
```

---

# Verify

Verify your environment:
```bash
$ source .venv/bin/activate
(.venv) $ python ./tools/verify_environment.py
```

Or, if you used `uv`, just do `uv run` (no need to manually activate the `.venv`):
```bash
$ uv run ./tools/verify_environment.py
```

Finally, run `pytest`. You should see something like this:
```bash
$ source .venv/bin/activate
(.venv) $ pytest -q
..................................................                         [100%]
52 passed in 12.01s
```

---

# Start the Labs

Congratulations! Now begin the course and start the Lab 0:
```bash
# You can also invoke these with `-h` to see help
$ ./startnextlab            # Starts unit1.lab0
$ ./whichlab                # Which lab are you in?
$ pytest --only-lab         # Run tests for this lab only
$ grep -r 'TODO-LAB' src    # Finds code for you to write
```

The course and lab guides are at https://ft4.ai.

---

## Feedback and PRs

Feedback greatly appreciated! PRs, suggestions, and any comments 
are all extremely helpful.
