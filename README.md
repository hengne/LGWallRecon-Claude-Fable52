# LGWallRecon

**Light-guide inner-wall reconstruction from an entrance-face scan.**
[中文说明](README.zh-CN.md) · [User manual (EN)](docs/manual/LGWallRecon_manual_en.pdf) · [用户手册 (中文)](docs/manual/LGWallRecon_manual_zh.pdf)

Author: Hengne Li (李衡讷), South China Normal University — <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>

A hollow conical light guide (entrance ⌀17.5 mm, exit ⌀8 mm, 100 mm long, mirror wall) is scanned with a small
diffuse light spot on its entrance face while a PMT reads the exit. An ideal cone would respond identically
everywhere; the measured map is not flat. LGWallRecon

1. simulates the guide with a specular ray tracer in which the wall is the CAD cone plus a free-form B-spline
   deviation field (CPU with numba, or GPU with OpenCL at about 5×10⁷ rays/s),
2. inverts the measured 100×100-point map for a wall shape that reproduces it,
3. checks itself against Geant4, a reciprocity identity and a closure test, and
4. produces zoomable vector figures and a single-file report page with a live 3-D ray animation.

![measured, simulated and residual maps](results/en/response_maps.svg)

## Result in one paragraph

A wall is found for which simulation and measurement differ by about 1 % (RMS) in the inner aperture, against
about 0.8 % point noise; an ideal cone leaves 5.4 %. The local unevenness needed is small: 13 µm RMS in the
published fit, on top of an axisymmetric narrowing of the bore.
**That wall is not unique.** In a closure test with a known wall the synthetic map is reproduced down to the
noise, but the recovered wall correlates with the true one at only about 0.1. The scan is sensitive to roughly
100 combinations of the wall shape; inside that visible subspace the recovery is good (correlation 0.7–0.97),
outside it nothing is learned. Read the wall figure as *the least unevenness that explains the scan*, not as a
survey of the real surface. Exact numbers are in `results/numbers.json`.

![wall figure](results/en/inner_surface.svg)

## Quick start

```bash
git clone git@github.com:hengne/LGWallRecon-Claude-Fable52.git && cd LGWallRecon-Claude-Fable52
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
./run_all.sh                  # aperture -> reconstruction -> figures + report  (~30 min with a GPU)
./run_all.sh --with-checks    # also closure test, reciprocity test and Geant4 cross-check
```

Outputs: `results/` (figures in `zh/` and `en/`, `wall_delta.csv`, `wall_coeffs.json`, `numbers.json`) and
`docs/LGWallRecon_report_{en,zh}.html` (open in a browser; they work offline).

Requirements: Python ≥ 3.10 with the packages in `requirements.txt`. Optional: an OpenCL GPU (`clinfo` must list
it; otherwise run with `LG_GPU=0`, about twenty times slower), Geant4 11.x for the cross-check, `node` for the
animated-ray SVG. Jobs run through `tools/run_limited.sh`, which caps memory with a systemd scope and adapts
the CPU share to the machine load; on a system without `systemd-run`, call the Python scripts directly.

## Step by step

All commands run inside `src/`, prefixed with `../tools/run_limited.sh` (abbreviated `R` below).

| Step | Command | What it does |
|---|---|---|
| 1 | `R python edge.py` and `R python edge_spot.py` | entrance centre, radius, ellipticity and light-spot shape from the sharp edge of the scan → `out/edge.npz` |
| 2 | `R ./pipeline.sh data` | the reconstruction chain below → `out/fitB_datab.npz`, `out/final_data.npz` |
| 2a | `stageA.py 10 fix 3.0` | axisymmetric wall profile from the radial response |
| 2b | `refine.py`, `kernel_modes.py` | linear response of the map to every wall coefficient (azimuthal Fourier modes) |
| 2c | `stageB.py` | damped least-squares inversion of the 2-D wall; every step is checked with a full ray trace |
| 2d | `final.py` | high-statistics simulated map, residuals, wall on a fine grid |
| 3 | `R ./closure.sh 1` | closure test and resolution study |
| 4 | `R python test_reciprocity.py ../out/fitB_datab.npz`, `../g4/run_final.sh ../out/fitB_datab.npz` | self-consistency and Geant4 cross-check |
| 5 | `R ./make_report.sh` | figures, numbers, wall exports, report pages |

The manual documents every script, its options, file formats, how to adapt the toolkit to another light guide,
and troubleshooting.

## Checks

| Check | Result |
|---|---|
| Geant4, ideal cone | T = 0.209 ± 0.002, flat; theory (R₂/R₁)² = 0.2090 |
| Geant4, fitted wall (720 000 facets) | Geant4 / tracer = 1.002 ± 0.001 |
| GPU against CPU tracer | identical ray by ray in double precision; single precision unbiased |
| Reciprocity (flux in → out equals out → in) | ratio 1.00 ± 0.01 |
| Closure test, map | reproduced down to the noise |
| Closure test, wall | **not recovered** (correlation ≈ 0.1; 0.7–0.97 inside the scan-visible subspace) |

## Layout

```
data/      measured scan (CSV) and CAD geometry (STEP)
src/       tracer (lgtrace.py CPU, lgcl.py GPU), reconstruction chain, tests, studies, figure and report builders
g4/        Geant4 cross-check (lgsim.cc)
web/       browser tracer, report template, language tables (i18n/zh.json, i18n/en.json)
tools/     run_limited.sh resource wrapper
results/   figures (zh/, en/), wall tables, numbers
docs/      single-file report pages and the user manuals
```

## Limits

The wall is not unique (see above); the azimuthal-mean profile is degenerate with the signal scale and the exit
radius; everything is attributed to geometry (perfect mirror, uniform PMT, Lambertian source); a 0.4 mm ring at
the entrance rim is not fitted; resolution is 2.5 mm along the axis and 15° in azimuth. To pin down the real
wall, add independent measurements: collimated-beam scans at several angles, angle-resolved readout, a scan
from the exit side, or profilometry.

## Repositories

GitHub (public): <https://github.com/hengne/LGWallRecon-Claude-Fable52> · GitLab (SCNU group): <https://gitlab.cern.ch/SCNU/LGWallRecon-Claude-Fable52>

## License

MIT, see [LICENSE](LICENSE).

## Citing

Hengne Li, *LGWallRecon: light-guide inner-wall reconstruction from an entrance-face scan* (2026).
