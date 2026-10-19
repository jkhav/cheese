#!/bin/sh
# One-shot setup: check the NVIDIA driver, find (or install, without sudo) a CUDA compiler, build, run the tests.
#
#   ./install.sh              GPU build if a driver is present (CPU build too), then tests/check.sh
#   ./install.sh --cpu-only   CPU build only (any machine)
#   NQ="1 4" ./install.sh     compile 1-4 qubit support (default "2 3": shorter nvcc build)
#
# CUDA compiler, first found: $NVCC, nvcc on PATH, /usr/local/cuda/bin/nvcc, else NVIDIA's pip wheels
# (nvidia-cuda-nvcc / -runtime / -cccl) in ./.venv matching the driver's CUDA major version - no root needed.
# The NVIDIA driver itself cannot be installed this way: without one, only the CPU build is made.
set -e
cd "$(dirname "$0")"
say() { printf '\n== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
CPU_ONLY=0; [ "$1" = "--cpu-only" ] && CPU_ONLY=1
export NQ=${NQ:-2 3}

say "prerequisites"
command -v python3 >/dev/null || die "python3 not found"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 8))' || die "python >= 3.8 needed"
command -v g++ >/dev/null || die "g++ not found (install your distribution's build tools)"
echo "python3 $(python3 -c 'import platform; print(platform.python_version())'), $(g++ --version | head -1)"

GPU=0
if [ $CPU_ONLY = 0 ] && command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  GPU=1
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
  DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -1)
  echo "GPU architecture sm_$CC, driver supports CUDA $DRV_CUDA"
elif [ $CPU_ONLY = 0 ]; then
  echo "no working NVIDIA driver (nvidia-smi): CPU build only"
fi

venv_pip() {   # NVIDIA's CUDA wheels into .venv
  [ -x .venv/bin/python ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q "$@"
}
find_in_venv() { find .venv -path "$1" 2>/dev/null | head -1; }

# ---- Thrust/CUB headers (CCCL): toolkit, or pip wheel ----
CCCL=${CCCL:-}
NVCC=${NVCC:-}
CUDA_INC=; CUDA_LIB=
if [ $GPU = 1 ]; then
  say "CUDA compiler"
  [ -z "$NVCC" ] && command -v nvcc >/dev/null && NVCC=$(command -v nvcc)
  [ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
  if [ -z "$NVCC" ]; then
    echo "no nvcc: installing NVIDIA's CUDA $DRV_CUDA compiler wheels into .venv (user space)"
    if [ "${DRV_CUDA:-0}" -ge 13 ]; then venv_pip nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-cccl
    elif [ "${DRV_CUDA:-0}" -ge 12 ]; then venv_pip nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12
    else die "driver supports CUDA $DRV_CUDA; update the NVIDIA driver (CUDA 12+ needed)"; fi
    NVCC=$(find_in_venv "*/bin/nvcc")
    [ -n "$NVCC" ] || die "nvcc wheel installed but nvcc not found under .venv"
    CUDA_INC=$(dirname "$(find_in_venv '*/include/cuda_runtime.h')")
    CUDA_LIB=$(dirname "$(find_in_venv '*/lib*/libcudart.so*')")
    [ -z "$CCCL" ] && CCCL=$(dirname "$(dirname "$(find_in_venv '*/thrust/version.h')")")   # .../thrust/version.h
  fi
  "$NVCC" --version | tail -2
fi
if [ -z "$CCCL" ] && [ $GPU = 0 ]; then   # CPU build: headers only
  for d in /usr/local/cuda/include /opt/cuda/include; do [ -f "$d/thrust/version.h" ] && CCCL=$d && break; done
  [ -z "$CCCL" ] && [ -f "$(find_in_venv '*/thrust/version.h')" ] && CCCL=$(dirname "$(dirname "$(find_in_venv '*/thrust/version.h')")")
  if [ -z "$CCCL" ]; then
    say "Thrust headers (pip nvidia-cuda-cccl into .venv)"
    venv_pip nvidia-cuda-cccl
    CCCL=$(dirname "$(dirname "$(find_in_venv '*/thrust/version.h')")")
  fi
fi

# ---- record the configuration for build.sh ----
mkdir -p build
cat > build/config.sh <<EOF
# written by install.sh; read by build.sh
NVCC='$NVCC'
CCCL='$CCCL'
CUDA_INC='$CUDA_INC'
CUDA_LIB='$CUDA_LIB'
ARCH='${CC:+sm_$CC}'
EOF
cat build/config.sh

say "build (qubit counts $NQ)"
./build.sh cpu
if [ $GPU = 1 ]; then
  echo "nvcc build: a few minutes per qubit count"
  ./build.sh cuda
fi

say "tests"
tests/check.sh
[ $GPU = 1 ] && echo "GPU binary: build/cudasynth. Timings: tests/bench.sh" || echo "CPU binary: build/cudasynth-cpu. Timings: tests/bench.sh"
