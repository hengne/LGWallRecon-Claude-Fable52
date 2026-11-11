#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# The reconstruction chain: axisymmetric profile -> linear-response kernels -> 2-D wall inversion (two rounds)
# -> high-statistics comparison with the scan.
#   usage: pipeline.sh <tag>            results: ../out/fitB_<tag>b.npz, ../out/final_<tag>.npz
#   LG_SYNTH=<npz>  run on a synthetic scan (closure test) instead of the measured one
# Run it through ../tools/run_limited.sh (resource limits, GPU on).  edge.py and edge_spot.py must have been run before.
set -e
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}; T=$1
LG_TAG=$T $PY -u stageA.py 10 fix 3.0 | tail -2
$PY refine.py ../out/A_$T.npz 40 ../out/start_$T.npz
$PY kernel_modes.py ../out/start_$T.npz 24 ${T}a 12000 0.06 | tail -1
LG_MU=3000 $PY -u stageB.py ${T}a ${T}a 1.0 40 ../out/start_$T.npz | grep "^it" | sed -n '1p;$p'
$PY kernel_modes.py ../out/fitB_${T}a.npz 24 ${T}b 12000 0.06 | tail -1
LG_MU=20000 $PY -u stageB.py ${T}b ${T}b 1.0 40 ../out/fitB_${T}a.npz | grep "^it" | sed -n '1p;$p'
$PY final.py ../out/fitB_${T}b.npz $T 40000 | tail -6
