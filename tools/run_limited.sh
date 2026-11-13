#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
# Run a command under resource limits, so that a simulation job cannot exhaust the memory of a shared
# machine and yields the CPU to everything else:
#   - transient systemd user scope: MemoryMax (the job is killed above it), no swap, CPUQuota
#   - nice 19 / ionice idle / CPUWeight 10  (other jobs win the CPU)
#   - CPU count adapts to how busy the machine is when the job starts:
#       cores = (all cores) - (cores busy with other work) - 4 spare, clamped to [2, 24]
# The limits apply to one invocation.  Jobs started side by side each get their own scope, so start
# parallel work from inside ONE invocation (as g4/run_final.sh does) if a total cap is wanted.
# usage: tools/run_limited.sh <command> [args...]
#        tools/run_limited.sh --cores          print the number of cores a job would get now, and exit
# override: LG_MEM=6G LG_CPUS=8 tools/run_limited.sh ...
MEM=${LG_MEM:-6G}
if [ -z "$LG_CPUS" ]; then
    NPROC=$(nproc)
    # CPU actually in use right now (1 s sample of /proc/stat), more reliable than the load average
    read -r _ u1 n1 s1 i1 w1 _ < /proc/stat; sleep 1; read -r _ u2 n2 s2 i2 w2 _ < /proc/stat
    BUSY=$(( ( (u2 + n2 + s2 - u1 - n1 - s1) * NPROC + (u2 + n2 + s2 + i2 + w2 - u1 - n1 - s1 - i1 - w1) / 2 ) / (u2 + n2 + s2 + i2 + w2 - u1 - n1 - s1 - i1 - w1) ))
    LG_CPUS=$(( NPROC - BUSY - 4 ))
    [ $LG_CPUS -gt 24 ] && LG_CPUS=24
    [ $LG_CPUS -lt 2 ] && LG_CPUS=2
    [ "$1" = "--cores" ] || echo "[run_limited] $BUSY of $NPROC cores busy -> using $LG_CPUS cores, MemoryMax $MEM" >&2
fi
CPUS=$LG_CPUS
if [ "$1" = "--cores" ]; then echo "$CPUS"; exit 0; fi
export LG_GPU=${LG_GPU:-1}        # wall tracing runs on the GPU (OpenCL) unless LG_GPU=0
export NUMBA_NUM_THREADS=$CPUS OMP_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS
exec systemd-run --user --scope --quiet --collect \
    -p MemoryMax=$MEM -p MemorySwapMax=0 -p CPUQuota=$((CPUS * 100))% -p CPUWeight=10 \
    nice -n 19 ionice -c 3 "$@"
