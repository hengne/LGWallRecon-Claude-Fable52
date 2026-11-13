#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# The reconstruction chain: axisymmetric profile -> linear-response kernels -> 2-D wall inversion (two rounds)
# -> high-statistics comparison with the scan.
#   usage: pipeline.sh <tag>            results: ../out/fitB_<tag>b.npz, ../out/final_<tag>.npz
#   LG_SYNTH=<npz>  run on a synthetic scan (closure test) instead of the measured one
#   LG_KAMP=<mm>    finite-difference amplitude of the response kernels (default 0.06; see results/checks/README.md)
# Run it through ../tools/run_limited.sh (resource limits, GPU on).  edge.py and edge_spot.py must have been run before.
# The files of an earlier run with the same tag are removed first and every step is checked, so a failed
# step stops the chain instead of being papered over by old files.  Full output: ../out/pipeline_<tag>.log
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}; T=$1; LOG=../out/pipeline_$T.log; KAMP=${LG_KAMP:-0.06}
rm -f ../out/A_$T.npz ../out/start_$T.npz ../out/kmodes_${T}a.npz ../out/kmodes_${T}b.npz \
      ../out/fitB_${T}a.npz ../out/fitB_${T}b.npz ../out/final_$T.npz
: > "$LOG"
run() {                                  # run one step, all output to the log; stop the chain if it fails
    echo "== $*" >> "$LOG"
    "$@" >> "$LOG" 2>&1 || { echo "pipeline.sh: step failed: $*" >&2; tail -n 25 "$LOG" >&2; exit 1; }
}
last() { tac "$LOG" | sed '/^== /q' | tac; }     # output of the latest step
run env LG_TAG=$T $PY -u stageA.py 10 fix 3.0;                                   last | tail -n 2
run $PY refine.py ../out/A_$T.npz 40 ../out/start_$T.npz
run $PY kernel_modes.py ../out/start_$T.npz 24 ${T}a 12000 $KAMP;               last | tail -n 1
run env LG_MU=3000 $PY -u stageB.py ${T}a ${T}a 1.0 40 ../out/start_$T.npz;      last | grep "^it" | sed -n '1p;$p'
run $PY kernel_modes.py ../out/fitB_${T}a.npz 24 ${T}b 12000 $KAMP;             last | tail -n 1
run env LG_MU=20000 $PY -u stageB.py ${T}b ${T}b 1.0 40 ../out/fitB_${T}a.npz;   last | grep "^it" | sed -n '1p;$p'
run $PY final.py ../out/fitB_${T}b.npz $T 40000;                                 last | tail -n 6
