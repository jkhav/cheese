#!/bin/sh
# Build both backends from the same source (install.sh runs this; it writes build/config.sh first).
#   ./build.sh cpu    g++ + OpenMP (Thrust's OMP backend): any machine, for development and tests
#   ./build.sh cuda   nvcc for the GPU
# Settings (environment, else build/config.sh): NVCC, ARCH (sm_86 RTX 30xx, sm_89 RTX 40xx, sm_120 RTX 50xx;
# default native), CCCL (Thrust headers dir), CUDA_INC / CUDA_LIB (pip-installed toolkit), NQ="2 3" (qubit-count
# range compiled in, default "1 4"; nvcc takes minutes per count, so narrow it).
set -e
cd "$(dirname "$0")"
mkdir -p build
_NVCC=$NVCC; _ARCH=$ARCH; _CCCL=$CCCL; _INC=$CUDA_INC; _LIB=$CUDA_LIB
[ -f build/config.sh ] && . build/config.sh
NVCC=${_NVCC:-${NVCC:-nvcc}}; ARCH=${_ARCH:-${ARCH:-native}}; CCCL=${_CCCL:-$CCCL}
CUDA_INC=${_INC:-$CUDA_INC}; CUDA_LIB=${_LIB:-$CUDA_LIB}
NQR=${NQ:-1 4}; NQDEF="-DNQ_MIN=${NQR% *} -DNQ_MAX=${NQR#* }"
case "${1:-cpu}" in
  cpu)
    CCCL=${CCCL:-$(dirname "$(dirname "$(find .venv /usr/local/cuda/include /opt/cuda/include -path '*/thrust/version.h' 2>/dev/null | head -1)")")}
    g++ -std=c++17 -O3 -march=native -fopenmp -x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP $NQDEF \
        -I"$CCCL" src/cudasynth.cu -o build/cudasynth-cpu
    echo build/cudasynth-cpu ;;
  cuda)
    FLAGS="-std=c++17 -O3 -arch=$ARCH --expt-relaxed-constexpr $NQDEF"
    [ -n "$CUDA_INC" ] && FLAGS="$FLAGS -I$CUDA_INC"
    [ -n "$CCCL" ] && FLAGS="$FLAGS -I$CCCL"
    [ -n "$CUDA_LIB" ] && FLAGS="$FLAGS -L$CUDA_LIB -Xlinker -rpath=$CUDA_LIB"
    if ! $NVCC $FLAGS src/cudasynth.cu -o build/cudasynth 2> build/nvcc.log; then
      if grep -q "unsupported GNU version" build/nvcc.log; then   # host g++ newer than this CUDA release
        echo "nvcc: host compiler newer than supported; retrying with -allow-unsupported-compiler"
        $NVCC $FLAGS -allow-unsupported-compiler src/cudasynth.cu -o build/cudasynth
      else cat build/nvcc.log >&2; exit 1; fi
    fi
    echo build/cudasynth ;;
esac
