#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# Build both user manuals (English and Chinese) from scratch:
#   numbers.tex from results/numbers.json, PNG figures from results/<lang>/*.svg, then xelatex twice.
# usage: build_manual.sh [github url] [gitlab url]
#   SKIP_FIGS=1   reuse the PNG figures already in fig/
#   PY=<python>   interpreter for make_numbers_tex.py (default: the package venv, else python3)
set -e
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
[ -x "$PY" ] || PY=python3
$PY make_numbers_tex.py "$1" "$2"
[ "$SKIP_FIGS" = 1 ] || ./make_manual_figs.sh
for lang in en zh; do
    doc=LGWallRecon_manual_$lang
    for pass in 1 2 3; do
        xelatex -interaction=nonstopmode -halt-on-error $doc.tex > $doc.build.log 2>&1 || { tail -30 $doc.build.log; echo "xelatex failed for $doc"; exit 1; }
    done
    grep -c "Overfull" $doc.log | sed "s/^/$doc: overfull boxes: /" || true
    rm -f $doc.aux $doc.log $doc.out $doc.toc $doc.lof $doc.lot $doc.build.log
    echo "$doc.pdf written"
done
