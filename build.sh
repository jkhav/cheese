#!/bin/sh
# Build both backends from the same source.
#   ./build.sh cpu    g++ + OpenMP (Thrust's OMP backend): any machine, for development and tests
#   ./build.sh cuda   nvcc for the GPU (set ARCH, e.g. ARCH=sm_86 for RTX 30xx, sm_89 RTX 40xx, sm_120 RTX 50xx)
# CCCL (Thrust) headers: from the CUDA toolkit, or set CCCL=/path/to/include/cccl (e.g. pip nvidia-cuda-cccl).
set -e
cd "$(dirname "$0")"
mkdir -p build
case "${1:-cpu}" in
  cpu)
    CCCL=${CCCL:-$(ls -d .venv/lib/python*/site-packages/nvidia/cu*/include/cccl 2>/dev/null | head -1)}
    g++ -std=c++17 -O3 -march=native -fopenmp -x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP \
        -I"$CCCL" src/cudasynth.cu -o build/cudasynth-cpu
    echo build/cudasynth-cpu ;;
  cuda)
    NVCC=${NVCC:-nvcc}
    $NVCC -std=c++17 -O3 -arch=${ARCH:-native} --expt-relaxed-constexpr ${CCCL:+-I"$CCCL"} \
        src/cudasynth.cu -o build/cudasynth
    echo build/cudasynth ;;
esac
