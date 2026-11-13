#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# Full reproduction, from data/scan_summary.csv to the figures and the report pages.
#   ./run_all.sh                 self-tests -> aperture -> reconstruction -> figures + report   (~45 min on the GPU)
#   ./run_all.sh --with-checks   also the closure test, kernel test, reciprocity, free reflectivity and the Geant4 cross-check
# Every job runs through tools/run_limited.sh (hard memory cap, CPU share adapted to the machine load,
# GPU on; set LG_GPU=0 for a CPU-only run, which is about twenty times slower).
# Any failing step or self-test stops the run.  The report lists a check only with the wall it was run on;
# checks left over from an earlier wall are marked as such.
set -euo pipefail
cd "$(dirname "$0")/src"
R=../tools/run_limited.sh; PY=../.venv/bin/python
CHECKS=0; [ "${1:-}" = "--with-checks" ] && CHECKS=1
mkdir -p ../out
LG_GPU=0 $R $PY test_sampler.py      # quasi-random rays fill their unit cube
LG_GPU=0 $R $PY test_grazing_hit.py  # no grazing wall hit is skipped
LG_GPU=0 $R $PY test_ideal_cone.py   # ideal cone: area-averaged transmission = (R2/R1)^2
$R $PY edge.py                       # entrance aperture from the sharp edge of the scan
$R $PY edge_spot.py                  # light-spot shape
$R ./pipeline.sh data                # axisymmetric profile, kernels, 2-D inversion, high-statistics map
$R $PY compare_walls.py ../out/final_data.npz ../results/earlier/wall_coeffs_cb81f08.json   # against the wall of an earlier fit of this scan
if [ "${LG_GPU:-1}" != 0 ]; then $R $PY test_gpu_vs_cpu.py ../out/fitB_datab.npz; fi
if [ $CHECKS = 1 ]; then
    $R ./closure.sh 1
    $R $PY -u test_jacobian.py datab ../out/fitB_datab.npz > ../out/test_jacobian.log        # linear kernels against exact finite changes of the map
    LG_GPU=0 $R $PY -u test_reciprocity.py ../out/fitB_datab.npz | tee ../out/recip.log
    LG_TAG=refl LG_START=../out/A_data.npz $R $PY -u stageA.py 10 refl 3.0 > ../out/A_refl.log   # free the mirror reflectivity, starting from the perfect-mirror fit
    if command -v geant4-config > /dev/null || [ -f ~/.local/opt/geant4/current/bin/geant4.sh ]; then ../g4/run_final.sh ../out/fitB_datab.npz; fi
fi
$R ./make_report.sh ../out/final_data.npz
echo "done: results/ and docs/LGWallRecon_report_{zh,en}.html"
