# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # or Debug / RelWithDebInfo
make -j$(nproc)
```

On GCC (Linux), the codegen examples require `-lstdc++fs` (already wired in `examples/CMakeLists.txt` via `CMAKE_COMPILER_IS_GNUCXX`). The build produces:

- `build/src/tinympc/libtinympcstatic.a` — the static library
- `build/examples/<name>` — 8 standalone example executables

Run an example directly (no arguments needed):

```bash
./build/examples/quadrotor_hovering
./build/examples/cartpole_example
./build/examples/codegen_cartpole   # writes generated code to disk
```

There are no formal tests. CI (`cmake-multi-platform.yml`) builds on ubuntu/macos with gcc and clang, then calls `ctest` (which currently has no registered tests).

## Code generation

Set `USING_CODEGEN ON` in the root `CMakeLists.txt` only when building the Python wrapper. It copies Eigen + TinyMPC sources into `build/codegen_src/` and installs them to `${CMAKE_INSTALL_DATAROOTDIR}/tinympc/codegen_files`. Python, Julia, and MATLAB language bindings live in separate repositories.

## Architecture

TinyMPC is an embedded-friendly MPC solver using ADMM with a fixed-step Riccati factorization cached at setup time. All numerics use Eigen with `double` (`tinytype`), `tinyMatrix` (dynamic Eigen matrix), and `tinyVector`.

### Data structures (`src/tinympc/types.hpp`)

| Struct | Role |
|--------|------|
| `TinySolver` | Root handle; holds pointers to all sub-structs |
| `TinyCache` | Precomputed infinite-horizon gains (`Kinf`, `Pinf`, `Quu_inv`, `AmBKt`) and adaptive-rho sensitivity matrices |
| `TinyWorkspace` | All per-iteration storage: primal (`x`, `u`), slack (`v`/`z`/`vc`/`zc`/`vl`/`zl` + TV variants), dual (`g`/`y` + variants), references (`Xref`, `Uref`), and constraint matrices |
| `TinySolution` | Output: state trajectory `x` (nx×N), control `u` (nu×N-1), `iter`, `solved` flag |
| `TinySettings` | Tolerances, `max_iter`, per-constraint enable flags, adaptive-rho params |

### Setup flow (`src/tinympc/tiny_api.cpp`)

1. `tiny_setup()` allocates and initializes all sub-structs.
2. `tiny_precompute_and_set_cache()` runs the **Riccati recursion** to compute `Kinf`, `Pinf`, `Quu_inv`, `AmBKt`, `APf`, `BPf` — these never change unless dynamics or costs change.
3. Constraint setters (`tiny_set_bound_constraints`, `tiny_set_cone_constraints`, `tiny_set_linear_constraints`, `tiny_set_tv_linear_constraints`) populate workspace matrices and flip the corresponding enable flag in settings.

### ADMM loop (`src/tinympc/admm.cpp`, `solve()` at line 331)

Each iteration, in order:

1. **`update_linear_cost()`** — forms cost vectors `q` and `r` from references, duals, and slacks.
2. **`backward_pass_grad()`** — Riccati backward sweep computing control perturbations `d` using cached gains.
3. **`forward_pass()`** — rolls out `u = -Kx - d`, `x⁺ = Ax + Bu + f`.
4. **`update_slack()`** — projects primal + dual onto each active constraint set (box → `cwiseMin/Max`; SOC → `project_soc()`; hyperplane → `project_hyperplane()`).
5. **`update_dual()`** — increments each dual variable by `primal - slack`.
6. **Termination check** (`check_termination` every N steps) — compares max absolute primal/dual residuals to tolerances.
7. **Optional adaptive-rho** — updates `rho` and recomputes cache via precomputed Taylor-expansion matrices (`C1`, `C2`, `dKinf_drho`, etc.).

### Constraint system

Constraints are **independent ADMM variable splits**, each with its own primal slack and dual. Adding a constraint type does not restructure the solver — it only populates the relevant workspace arrays and sets the enable flag. Time-varying linear constraints store all N horizon copies row-stacked (`num_*_linear × N` rows).

### Typical MPC loop pattern

```cpp
TinySolver *solver;
tiny_setup(&solver, Adyn, Bdyn, fdyn, Q.asDiagonal(), R.asDiagonal(), rho, nx, nu, N, verbose);
tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);
solver->work->Xref = reference_traj;      // set reference once or per step

for (int k = 0; k < horizon; k++) {
    tiny_set_x0(solver, x_current);
    tiny_solve(solver);
    u_applied = solver->solution->u.col(0);
}
```
