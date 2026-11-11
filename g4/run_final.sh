#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# Geant4 cross-check of a fitted wall: the wall becomes a tessellated cavity (1440 x 251 mesh, 7.2e5 facets),
# NPROC single-threaded lgsim jobs trace optical photons from 120 source points, and the result is compared
# with the ray tracer (-> ../out/g4_compare.json).  About 10 ms per photon; each job needs ~0.7 GB.
# usage: run_final.sh <fit npz> [NPROC = 12] [photons per point = 4000]
set -e
cd "$(dirname "$0")"
FIT=$(realpath "$1"); NPROC=${2:-12}; NPH=${3:-4000}
[ -f ~/.local/opt/geant4/current/bin/geant4.sh ] && source ~/.local/opt/geant4/current/bin/geant4.sh
[ -x build/lgsim ] || { mkdir -p build && (cd build && cmake .. > cmake.log && make -j4 > make.log); }
mkdir -p run && rm -f run/final_*.out run/final_pts_*.txt
(cd ../src && LG_GPU=0 ../.venv/bin/python export_mesh.py "$FIT" ../g4/run/mesh_final.txt ../g4/run/final_pts.txt 1440 251 120)
awk -v n=$NPROC '{ printf "%s\n", $0 > sprintf("run/final_pts_%02d.txt", (NR - 1) % n) }' run/final_pts.txt
for f in run/final_pts_??.txt; do
    i=${f#run/final_pts_}; i=${i%.txt}
    LG_CPUS=1 LG_MEM=2G ../tools/run_limited.sh build/lgsim run/mesh_final.txt $f $NPH run/final_$i.out -0.34 0.55 1.0 $((100 + 10#$i)) > run/final_$i.log 2>&1 &
done
wait
(cd ../src && ../tools/run_limited.sh ../.venv/bin/python g4_compare.py "$FIT" ../g4/run/final_??.out)
