#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# Full reproduction, from data/scan_summary.csv to the figures and the report pages.
#   ./run_all.sh                 aperture -> reconstruction -> figures + report        (~30 min on the GPU)
#   ./run_all.sh --with-checks   also the closure test, the reciprocity test and the Geant4 cross-check
# Every job runs through tools/run_limited.sh (hard memory cap, CPU share adapted to the machine load,
# GPU on; set LG_GPU=0 for a CPU-only run, which is about twenty times slower).
set -e
cd "$(dirname "$0")/src"
R=../tools/run_limited.sh; PY=../.venv/bin/python
mkdir -p ../out
$R $PY edge.py                       # entrance aperture from the sharp edge of the scan
$R $PY edge_spot.py                  # light-spot shape
$R ./pipeline.sh data                # axisymmetric profile, kernels, 2-D inversion, high-statistics map
if [ "$1" = "--with-checks" ]; then
    $R ./closure.sh 1
    LG_GPU=0 $R $PY -u test_reciprocity.py ../out/fitB_datab.npz | tee ../out/recip.log
    if command -v geant4-config > /dev/null || [ -f ~/.local/opt/geant4/current/bin/geant4.sh ]; then ../g4/run_final.sh ../out/fitB_datab.npz; fi
fi
$R ./make_report.sh ../out/final_data.npz
echo "done: results/ and docs/LGWallRecon_report_{zh,en}.html"
