#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# Geant4 cross-check of the tracer, with the source spot and the mirror reflectivity of the fit:
#   1. fitted wall: the wall becomes a tessellated cavity (1440 x 251 mesh, 7.2e5 facets); NPROC
#      single-threaded lgsim jobs trace optical photons from 120 source points   -> ../out/g4_compare.json
#   2. ideal cone (G4Cons), 30 source points along a radius                       -> ../out/g4_ideal.json
# About 10 ms per photon in the tessellated cavity; each job needs ~0.7 GB.  All jobs share ONE resource
# scope (tools/run_limited.sh): NPROC cores and NPROC + 2 GB in total.
# usage: run_final.sh <fit npz> [NPROC = idle cores, at most 12] [photons per point = 4000]
set -euo pipefail
cd "$(dirname "$0")"
FIT=$(realpath "$1"); NPH=${3:-4000}
NPROC=${2:-$(../tools/run_limited.sh --cores)}; [ "$NPROC" -gt 12 ] && NPROC=12
[ -f ~/.local/opt/geant4/current/bin/geant4.sh ] && source ~/.local/opt/geant4/current/bin/geant4.sh
[ -x build/lgsim ] || { mkdir -p build && (cd build && cmake .. > cmake.log && make -j4 > make.log); }
mkdir -p run && rm -f run/final_*.out run/final_pts_*.txt run/ideal.out ../out/g4_compare.json ../out/g4_ideal.json
(cd ../src && LG_GPU=0 ../.venv/bin/python export_mesh.py "$FIT" ../g4/run/mesh_final.txt ../g4/run/final_pts.txt 1440 251 120)
read -r SPOT SMAX REFL < run/mesh_final.txt.params
awk -v n="$NPROC" '{ printf "%s\n", $0 > sprintf("run/final_pts_%02d.txt", (NR - 1) % n) }' run/final_pts.txt
awk 'BEGIN { for (i = 0; i < 30; i++) printf "%.5f 0\n", 0.14 + i * 0.28 }' > run/ideal_pts.txt
export SPOT SMAX REFL NPH
LG_CPUS=$NPROC LG_MEM=$((NPROC + 2))G ../tools/run_limited.sh bash -c '
    pids=()
    for f in run/final_pts_??.txt; do
        i=${f#run/final_pts_}; i=${i%.txt}
        build/lgsim run/mesh_final.txt $f $NPH run/final_$i.out $SPOT $SMAX $REFL $((100 + 10#$i)) > run/final_$i.log 2>&1 &
        pids+=($!)
    done
    st=0; for p in "${pids[@]}"; do wait "$p" || st=1; done
    build/lgsim ideal run/ideal_pts.txt 40000 run/ideal.out $SPOT 0.7 1.0 7 > run/ideal.log 2>&1 || st=1
    exit $st' || { echo "run_final.sh: a Geant4 job failed, see g4/run/*.log" >&2; exit 1; }
NPTS=$(wc -l < run/final_pts.txt); NOUT=$(cat run/final_??.out | wc -l)
[ "$NOUT" -eq "$NPTS" ] || { echo "run_final.sh: $NOUT of $NPTS source points came back" >&2; exit 1; }
echo "$SPOT 0.7 1.0" > run/ideal.params
(cd ../src && LG_GPU=0 ../tools/run_limited.sh ../.venv/bin/python g4_compare.py "$FIT" ../g4/run/mesh_final.txt.params ../g4/run/final_??.out)
(cd ../src && LG_GPU=0 ../tools/run_limited.sh ../.venv/bin/python g4_compare.py ideal ../g4/run/ideal.params ../g4/run/ideal.out)
