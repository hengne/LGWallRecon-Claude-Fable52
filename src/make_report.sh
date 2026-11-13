#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# Everything that is shown to people, from a finished reconstruction:
#   results/final.npz, wall_delta.csv, wall_coeffs.json, numbers.json, {zh,en}/*.svg
#   docs/LGWallRecon_report_{zh,en}.html   (single-file pages with the live 3-D demonstration)
# usage: make_report.sh [final npz = ../out/final_data.npz] [github url] [gitlab url]
#   the URLs printed in the pages default to the home of the toolkit (override: arguments, or LG_GITHUB_URL / LG_GITLAB_URL)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}; FIN=${1:-../out/final_data.npz}
GH=${2:-${LG_GITHUB_URL:-https://github.com/hengne/LGWallRecon-Claude-Fable52}}
GL=${3:-${LG_GITLAB_URL:-https://gitlab.cern.ch/SCNU/LGWallRecon-Claude-Fable52}}
export LG_GPU=0                                   # nothing here needs the GPU
mkdir -p ../results/zh ../results/en ../docs
cp "$FIN" ../results/final.npz
$PY export_wall.py "$FIN" ../results
for lg in zh en; do
    $PY make_figures.py "$FIN" ../results/$lg $lg
    if command -v node > /dev/null; then $PY make_ray_svg.py "$FIN" ../results/$lg $lg; fi
    # closure-test figure and numbers, if the closure test was run (./closure.sh 1); otherwise the tracked ones stay
    if [ -f ../out/final_cl1.npz ] && [ -f ../out/synth_cl1.npz ]; then $PY make_closure_fig.py ../out/synth_cl1.npz ../out/final_cl1.npz ../results/$lg $lg; fi
done
$PY collect_numbers.py "$FIN" ../results
for lg in zh en; do
    $PY build_page.py "$FIN" ../results ../docs/LGWallRecon_report_$lg.html $lg "$GH" "$GL"
done
