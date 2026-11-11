#!/bin/bash
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# Render the SVG figures of results/<lang>/ to PNG (2x pixel density) for the PDF manuals.
# The SVGs use CSS custom properties, which rsvg-convert does not support, so headless Firefox
# takes the pictures.  Each SVG is shown through a small wrapper page that scales it by SCALE,
# because headless screenshots ignore the device-pixel-ratio preference.
# A private browser profile is kept in .ffprofile/ (not versioned).
# usage: make_manual_figs.sh
set -e
cd "$(dirname "$0")"
HERE=$PWD
RES=$(cd ../../results && pwd)
PROF=$HERE/.ffprofile
SCALE=2
mkdir -p "$PROF"
shot() {   # shot <lang> <name> <width> <height>   (viewBox size of the SVG, in CSS pixels)
    local lang=$1 name=$2 w=$3 h=$4
    local src=$RES/$lang/$name.svg out=$HERE/fig/$lang/$name.png wrap=$HERE/fig/.wrap_${lang}_$name.html
    [ -f "$src" ] || { echo "missing $src (skipped)"; return 0; }
    mkdir -p "$HERE/fig/$lang"
    rm -f "$out"
    printf '<!doctype html><meta charset="utf-8"><body style="margin:0;background:#fff"><img src="file://%s" width="%d" height="%d" style="display:block">\n' \
        "$src" $((w * SCALE)) $((h * SCALE)) > "$wrap"
    timeout 180 firefox --headless --no-remote --profile "$PROF" --window-size=$((w * SCALE)),$((h * SCALE)) \
        --screenshot "$out" "file://$wrap" > /dev/null 2>&1 || true
    rm -f "$wrap"
    [ -s "$out" ] && echo "fig/$lang/$name.png" || echo "FAILED: $name ($lang)"
}
for lang in en zh; do
    shot $lang inner_surface 1200 962
    shot $lang response_maps 1200 814
    shot $lang closure 1200 560
done
