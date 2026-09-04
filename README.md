# Sparse Variable Projection in Robotic Perception: Exploiting Separable Structure for Efficient Nonlinear Optimization

This is the code for the experiments in the paper Sparse Variable Projection in Robotic Perception: Exploiting SeparableStructure for Efficient Nonlinear Optimization by [Alan Papalia](https://alanpapalia.github.io/), [Nikolas Sanderson](https://pandahood.github.io/NikolasSanderson.github.io/), Haoyu Han, [Heng Yang](https://hankyang.seas.harvard.edu/), Hanumant Singh, and [Michael Everett](https://mfe7.github.io/). This was done in collaboration with UMich's [Robotic Exploration Lab](https://robex.engin.umich.edu/), Northeastern Field Robotics Group, [Northeastern Autonomy and Intelligence Labratory](https://neu-autonomy.github.io/lab_website/), and Harvard's [Computational Robotics Group](https://computationalrobotics.seas.harvard.edu/). For the rest of the GTSAM experiments please see this [repository](https://github.com/UMich-RobotExploration/varProj-gtsam).

You can find the paper [here](https://arxiv.org/abs/2512.07969)


If you use this work in your research, please cite:

```bibtex
@misc{papalia2025sparsevariableprojectionrobotic,
      title        = {Sparse Variable Projection in Robotic Perception: Exploiting Separable Structure for Efficient Nonlinear Optimization},
      author       = {Alan Papalia and Nikolas Sanderson and Haoyu Han and Heng Yang and Hanumant Singh and Michael Everett},
      year         = {2025},
      eprint       = {2512.07969},
      archivePrefix= {arXiv},
      primaryClass = {cs.RO},
      url          = {https://arxiv.org/abs/2512.07969},
}
```

## Getting started

The code builds as a CMake project. The following steps have been verified on
Ubuntu 22.04.

**Step 1:** Install dependencies

```
$ sudo apt-get install build-essential cmake libeigen3-dev liblapack-dev libblas-dev libsuitesparse-dev
```

**Step 2:** Clone the repository

```
$ git clone https://github.com/UMich-RobotExploration/variable-projection.git
```

**Step 3:** Initialize Git submodules

```
$ cd variable-projection
$ git submodule init
$ git submodule update
```

**Step 4:** Create build directory

```
$ mkdir build
```

**Step 5:** Configure build and generate Makefiles

```
$ cd build && cmake ..
```

**Step 6:** Build code

```
$ make -j
```

**Step 7:** Run the standard benchmark on the included data

```
$ ./bin/paper_experiments
```

**Step 8:** Install NumPy for the Python experiment runners

```
$ cd .. && python3 -m venv .venv && .venv/bin/pip install numpy
```

## Running the experiments

Three files at the top level of `examples/` drive the three experiments. Each
writes results as JSON; analysis and plotting are not included.

**Standard benchmark.** The C++ binary is the runner, configured by
`examples/config.json` (data path, rank range, number of inits, which
formulations to sweep). It sweeps every directory containing a `.pyfg` under
`abs_data_path`.

```
$ build/bin/paper_experiments                      # uses examples/config.json
$ build/bin/paper_experiments path/to/config.json  # or an explicit config
```

**Sphere sweep.** Generates the synthetic g2o-style sphere datasets, then
sweeps them.

```
$ .venv/bin/python examples/sphere_sweep.py generate --axis pgo
$ .venv/bin/python examples/sphere_sweep.py run --num-inits 25
$ .venv/bin/python examples/sphere_sweep.py aggregate
```

**Robust loss.** Sweeps the CosmoBench and Nebula datasets with a
Geman-McClure kernel under graduated non-convexity, from an odometry
initialization, and scores ATE against ground truth.

```
$ .venv/bin/python examples/run_cosmobench_irls_gnc_sweep.py \
      --out examples/data/analysis/cosmobench_irls_gnc_full
```

`examples/runners/` holds the remaining data-collection scripts (grid3D and SfM
sweeps, the GTSAM baselines, outlier injection, precompute and peak-RAM
measurement). Run any of them with `--help`.

## Notes

- **GPU.** Configure with `-DENABLE_GPU=ON` to build the CUDA solver and
  `gpu_paper_experiments`, which takes the same config as `paper_experiments`.
  Requires CUDAToolkit, and uses cuDSS if it can find it.
- **GTSAM baseline.** `gtsam_gnc_pgo` builds only if `find_package(GTSAM)`
  succeeds, and needs `-DENABLE_VECTORIZATION=OFF`. The default `-march=native`
  changes Eigen's alignment relative to a stock `libgtsam` and corrupts memory
  at runtime rather than failing to link, so build it in a separate build
  directory. The remaining GTSAM experiments live in
  [varProj-gtsam](https://github.com/UMich-RobotExploration/varProj-gtsam).
- **Shared config.** `sphere_sweep.py run` rewrites `examples/config.json` in
  place to retarget it at the sweep, and restores it afterwards. If the run is
  killed hard, restore it from `examples/config.json.swap-backup`.
- **Dense formulation.** `Dense` forms the `p x p` reduced system explicitly and
  is opt-in everywhere, gated by `max_dense_gb` in the config and by
  `--include-dense` in the robust sweep.
- **Environment overrides.** `VARPRO_FORMULATION` (comma-separated) isolates one
  formulation per process; `VARPRO_MAX_ITERATIONS` and
  `VARPRO_MAX_COMPUTATION_TIME` cap the solver (`<= 0` means no limit).
