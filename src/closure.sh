#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# Closure test: a KNOWN wall (measured axisymmetric profile + 7 bumps and dents) -> synthetic scan with the
# measured noise -> the same reconstruction chain -> comparison, plus the resolution study (which wall
# patterns the scan determines).  The chain is given the aperture of the synthetic wall, as it is given the
# measured aperture for the real scan.  Needs a finished `pipeline.sh data` (for ../out/A_data.npz).
# usage: closure.sh [seed = 1]        results: ../out/final_cl<seed>.npz, ../out/resolution.json
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}; SEED=${1:-1}; T=cl$SEED
rm -f ../out/synth_$T.npz ../out/resolution.json
$PY make_synth.py ../out/A_data.npz ../out/synth_$T.npz $SEED
export LG_SYNTH=../out/synth_$T.npz
./pipeline.sh $T
LG_GPU=0 $PY study_resolution.py ${T}b ../out/synth_$T.npz ../out/fitB_${T}b.npz
