# Kernel test logs

Output of `src/test_jacobian.py` (linear response kernels against exact finite changes of the map) for the
published wall, with two finite-difference amplitudes of `src/kernel_modes.py`:

- `test_jacobian_amp060.log`: 60 µm, the amplitude used by `src/pipeline.sh` for the published result
  (`kernel_modes.py ../out/fitB_dataa.npz 24 datab 12000 0.06`).
- `test_jacobian_amp015.log`: 15 µm, a study run (`kernel_modes.py ../out/fitB_dataa.npz 24 ktest 12000 0.015`,
  then `test_jacobian.py ktest ../out/fitB_datab.npz`). Selectable in the pipeline with `LG_KAMP=0.015`.

The smaller amplitude is somewhat more accurate everywhere, but neither predicts axisymmetric (mean-profile)
changes well. See the manual, section on the linear response.
