#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
# Run a command under hard resource limits so a simulation job can never hurt the (shared) machine.
#   - transient systemd user scope: MemoryMax (hard kill above it), no swap, CPUQuota
#   - nice 19 / ionice idle  (other users' jobs always win the CPU)
#   - CPU count adapts to how busy the machine is when the job starts:
#       cores = (all cores) - (cores busy with other work) - 4 spare, clamped to [4, 24]
# usage: tools/run_limited.sh <command> [args...]
# override: LG_MEM=6G LG_CPUS=8 tools/run_limited.sh ...
MEM=${LG_MEM:-6G}
if [ -z "$LG_CPUS" ]; then
    NPROC=$(nproc)
    # CPU actually in use right now (1 s sample of /proc/stat), more reliable than the load average
    read -r _ u1 n1 s1 i1 w1 _ < /proc/stat; sleep 1; read -r _ u2 n2 s2 i2 w2 _ < /proc/stat
    BUSY=$(( ( (u2 + n2 + s2 - u1 - n1 - s1) * NPROC + (u2 + n2 + s2 + i2 + w2 - u1 - n1 - s1 - i1 - w1) / 2 ) / (u2 + n2 + s2 + i2 + w2 - u1 - n1 - s1 - i1 - w1) ))
    LG_CPUS=$(( NPROC - BUSY - 4 ))
    [ $LG_CPUS -gt 24 ] && LG_CPUS=24
    [ $LG_CPUS -lt 4 ] && LG_CPUS=4
    echo "[run_limited] $BUSY of $NPROC cores busy -> using $LG_CPUS cores, MemoryMax $MEM" >&2
fi
CPUS=$LG_CPUS
export LG_GPU=${LG_GPU:-1}        # wall tracing runs on the GPU (OpenCL) unless LG_GPU=0
export NUMBA_NUM_THREADS=$CPUS OMP_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS
exec systemd-run --user --scope --quiet --collect \
    -p MemoryMax=$MEM -p MemorySwapMax=0 -p CPUQuota=$((CPUS * 100))% -p CPUWeight=10 \
    nice -n 19 ionice -c 3 "$@"
