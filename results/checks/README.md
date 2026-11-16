# Kernel test logs

Output of `src/test_jacobian.py` (linear response kernels against exact finite changes of the map) for the
published wall, with two finite-difference amplitudes of `src/kernel_modes.py`:

- `test_jacobian_amp060.log`: 60 µm, the amplitude used by `src/pipeline.sh` for the published result
  (`kernel_modes.py ../out/fitB_dataa.npz 24 datab 12000 0.06`).
- `test_jacobian_amp015.log`: 15 µm, a study run (`kernel_modes.py ../out/fitB_dataa.npz 24 ktest 12000 0.015`,
  then `test_jacobian.py ktest ../out/fitB_datab.npz`). Selectable in the pipeline with `LG_KAMP=0.015`.

The smaller amplitude is somewhat more accurate everywhere, but neither predicts axisymmetric (mean-profile)
changes well. See the manual, section on the linear response.

## Full rerun with the 15 µm amplitude (`kamp015_*`)

`LG_KAMP=0.015 ./run_all.sh --with-checks`, the whole chain and all checks with the smaller amplitude,
run after the result was published. It did not give a better fit, so the published result and the default
amplitude (60 µm) are unchanged.

| | published (60 µm) | rerun (15 µm) |
|---|---|---|
| data − simulation, r < 8 mm | 0.90 % | 0.91 % |
| χ²/N | 1.27 | 1.30 |
| local unevenness (RMS) | 12.9 µm | 12.1 µm |
| kernel test, non-axisymmetric | 0.84–0.99 | 0.87–1.00 |
| kernel test, gentle axisymmetric bump | 0.44–0.50 | 0.56–0.70 |
| kernel test, axisymmetric step | 0.13–0.27 | 0.15–0.42 |
| closure test, correlation of true and recovered wall | 0.18 | 0.16 |
| Geant4 / own tracer | 1.0007 ± 0.0015 | 0.9995 ± 0.0015 |

The two walls are close to each other: local unevenness correlated at 0.89 (0.96 in the first 20 mm, 0.76 in
the last 20 mm), difference 6 µm RMS, mean profiles 3 µm RMS apart, simulated maps 0.19 % apart. Both runs
start from the same axisymmetric profile and use the same method, so this agreement says that the chain is
reproducible, not that the wall is determined (the wall of an earlier version of the tracer, in
`results/earlier/`, fits the scan as well and is very different).

Starting from the published wall with 15 µm kernels computed around it, none of eight trial steps lowered
the objective (`kamp015_continue_published.log`). The finite-difference amplitude of the kernels is
therefore not what limits the fit.

- `kamp015_numbers.json`: `results/numbers.json` of the rerun
- `kamp015_wall_coeffs.json`: its wall, same format as `results/wall_coeffs.json`
- `kamp015_vs_published.json`: `compare_walls.py` of the rerun against the published wall
- `kamp015_pipeline.log`, `kamp015_test_jacobian.log`: logs of the fit and of the kernel test
- `kamp015_continue_published.log`: `stageB.py` from the published wall with the 15 µm kernels
