# TinyMPC → Vivado HLS FPGA Porting Plan

## Context

TinyMPC is a C++ ADMM-based MPC solver using Eigen dynamic matrices. Dynamic allocation,
runtime-sized loops, and C++ exceptions are not synthesizable by Vivado HLS. The goal is
a fully pipelined HLS implementation that maps cleanly onto UltraScale+ resources (DSP58E2,
BRAM36, URAM288, CARRY8, SRL32) using float (32-bit), with problem dimensions fixed at
compile time via C++ templates. The HLS core exposes an AXI4-Lite control interface and
BRAM ports for problem data / solution readback.

All new code lives under `src/tinympc_hls/` — no existing files are modified.

---

## Files to Create

```
src/tinympc_hls/
  types_hls.hpp          # Templated fixed-size structs replacing TinyWorkspace / TinyCache
  admm_hls.hpp           # ADMM kernel declarations + HLS pragmas
  admm_hls.cpp           # Kernel implementations
  tiny_api_hls.hpp       # Top-level HLS entry point (extern "C" for csim compat)
  testbench/
    tb_tinympc_hls.cpp   # C-sim testbench (drives quadrotor_hovering problem)
  tcl/
    create_project.tcl   # Vivado HLS Tcl project script (UltraScale+, float, 200 MHz)
```

---

## Step 1 — Templated Types (`types_hls.hpp`)

Replace all `tinyMatrix` (dynamic Eigen) with fixed-size C arrays. Template parameters
`NX`, `NU`, `NH` (horizon) become compile-time constants HLS can unroll against.

```cpp
template<int NX, int NU, int NH>
struct TinyCacheHLS {
    float Kinf[NU][NX];       // feedback gain  (NU×NX)
    float Pinf[NX][NX];       // value function hessian
    float Quu_inv[NU][NU];    // inverse input cost hessian
    float AmBKt[NX][NX];      // (A - B*Kinf)^T
    float APf[NX];            // A * Pinf * f  (affine term)
    float BPf[NU];            // B^T * Pinf * f
    float rho;
};

template<int NX, int NU, int NH>
struct TinyWorkspaceHLS {
    // Primal
    float x[NX][NH];          // state trajectory
    float u[NU][NH-1];        // input trajectory
    // Backward pass
    float d[NU][NH-1];        // feedforward corrections
    float p[NX][NH];          // cost-to-go gradient
    // Cost vectors
    float q[NX][NH];
    float r[NU][NH-1];
    // References
    float Xref[NX][NH];
    float Uref[NU][NH-1];
    // Bound slacks + duals (state & input)
    float v[NX][NH];   float vnew[NX][NH];   float g[NX][NH];
    float z[NU][NH-1]; float znew[NU][NH-1]; float y[NU][NH-1];
    // Dynamics (stored locally — loaded once from BRAM at start)
    float Adyn[NX][NX];
    float Bdyn[NX][NU];
    float fdyn[NX];
    // Cost weights
    float Q[NX];   // diagonal stored as vector
    float R[NU];
    // Initial state
    float x0[NX];
};

struct TinySettingsHLS {
    float abs_pri_tol;
    float abs_dua_tol;
    int   max_iter;
    int   check_termination;
    int   en_state_bound;
    int   en_input_bound;
};

template<int NX, int NU, int NH>
struct TinySolutionHLS {
    float x[NX][NH];
    float u[NU][NH-1];
    int   iter;
    int   solved;
};
```

**Key `ARRAY_PARTITION` rules** (in `admm_hls.cpp`):
- `x`, `u`, `vnew`, `znew`, `g`, `y`, `q`, `r`: `dim=1 complete` → all rows in registers,
  parallel element access across NX/NU per cycle
- `Kinf`, `Quu_inv`, `AmBKt`: `complete` → full register file, all elements readable in 1 cycle
- `Xref`, `Uref`, `Adyn`, `Bdyn`: `dim=2 cyclic factor=4` → balance BRAM ports

---

## Step 2 — ADMM Kernels (`admm_hls.cpp`)

Implement each sub-step as a separate function. HLS `DATAFLOW` runs independent sub-steps
as concurrent processes sharing ping-pong buffers.

### 2a. `update_linear_cost<NX,NU,NH>()`
```
q[:,k] = -Q .* Xref[:,k] - rho*(vnew[:,k] - g[:,k])
r[:,k] = -R .* Uref[:,k] - rho*(znew[:,k] - y[:,k])
```
- Fully data-parallel over k and element index → `UNROLL` inner, `PIPELINE II=1` outer
- Maps to DSP58E2 FMA chain: `D*A + B` where D=-Q, A=Xref, B=-rho*(vnew-g)
  Use DSP58 pre-adder (`D - A` path) for `vnew - g` with zero extra LUTs

### 2b. `backward_pass_grad<NX,NU,NH>()`
```
for k = NH-2 downto 0:
    d[:,k] = Quu_inv * (Bdyn^T * p[:,k+1] + r[:,k]) + BPf
    p[:,k] = q[:,k] + AmBKt * p[:,k+1] + APf
```
- Sequential over k (loop-carried dep on p), but inner GEMV pipelined
- `#pragma HLS PIPELINE II=1` on inner accumulation loop
- `Quu_inv` and `AmBKt` are complete-partitioned → parallel MAC tree across NU/NX rows

### 2c. `forward_pass<NX,NU,NH>()`
```
x[:,0] = x0
for k = 0 to NH-2:
    u[:,k] = -Kinf * x[:,k] - d[:,k]
    x[:,k+1] = Adyn * x[:,k] + Bdyn * u[:,k] + fdyn
```
- Sequential over k (loop-carried dep on x), inner GEMV pipelined
- `Kinf` complete-partitioned → NU parallel dot products in 1 loop iteration
- **SRL32 usage**: Buffer intermediate `p` values between backward and forward pass using
  SRL32 (Shift Register LUTs) instead of explicit FFs — saves ~40% FF on NX×NH pipeline delay

### 2d. `update_slack<NX,NU,NH>()`
Box projection (state and input):
```
vnew[:,k] = clip(x[:,k] + g[:,k], x_min, x_max)
znew[:,k] = clip(u[:,k] + y[:,k], u_min, u_max)
```
- **Unconventional: CARRY8 chains for min/max** — HLS maps float comparators to LUTs by default.
  Use `hls::min()` / `hls::max()` intrinsics which Vivado maps to CARRY8 fast comparators,
  not DSPs. Annotate with `#pragma HLS BIND_OP variable=vnew op=fmin impl=fabric` to force
  LUT/CARRY mapping and keep DSPs free for MACs.
- Fully parallel over k and element → `UNROLL factor=NH` on k loop

### 2e. `update_dual<NX,NU,NH>()`
```
g[:,k] += x[:,k] - vnew[:,k]
y[:,k] += u[:,k] - znew[:,k]
```
- Fully parallel, maps to DSP58 pre-adder: `g = g + (x - vnew)` uses `D - A + C` path
- `PIPELINE II=1`, `UNROLL`

### 2f. `check_termination<NX,NU,NH>()`
Compute max absolute primal/dual residual, compare to tolerances.
- Use tree-reduction with `UNROLL` for max-abs across NX/NU/NH
- Returns `int solved`

---

## Step 3 — Top-Level Entry Point (`tiny_api_hls.hpp`)

```cpp
template<int NX, int NU, int NH>
void tinympc_solve(
    TinyCacheHLS<NX,NU,NH>    *cache,      // #pragma HLS INTERFACE bram
    TinyWorkspaceHLS<NX,NU,NH> *work,      // #pragma HLS INTERFACE bram
    TinySettingsHLS            *settings,  // #pragma HLS INTERFACE s_axilite
    TinySolutionHLS<NX,NU,NH>  *solution,  // #pragma HLS INTERFACE bram
    int                        *status     // #pragma HLS INTERFACE s_axilite
);
```

AXI4-Lite bundle `ctrl` carries: `settings`, `status`, `ap_start`, `ap_done`, `ap_idle`.
BRAM ports carry: `cache` (read-only, initialized by PS/host), `work` (read-write),
`solution` (write-only output).

Outer ADMM iteration loop runs on-chip; host only writes problem data, asserts `ap_start`,
polls `ap_done`, then reads solution BRAM.

---

## Step 4 — Vivado HLS Pragma Summary

```cpp
// admm_hls.cpp — top of tinympc_solve()

// Interface
#pragma HLS INTERFACE s_axilite port=return   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=settings bundle=ctrl
#pragma HLS INTERFACE s_axilite port=status   bundle=ctrl
#pragma HLS INTERFACE bram      port=cache
#pragma HLS INTERFACE bram      port=work
#pragma HLS INTERFACE bram      port=solution

// Trajectory storage — dim=1 complete → NX/NU parallel access per cycle
#pragma HLS ARRAY_PARTITION variable=work->x    dim=1 complete
#pragma HLS ARRAY_PARTITION variable=work->u    dim=1 complete
#pragma HLS ARRAY_PARTITION variable=work->vnew dim=1 complete
#pragma HLS ARRAY_PARTITION variable=work->znew dim=1 complete
#pragma HLS ARRAY_PARTITION variable=work->g    dim=1 complete
#pragma HLS ARRAY_PARTITION variable=work->y    dim=1 complete
#pragma HLS ARRAY_PARTITION variable=work->d    dim=1 complete
#pragma HLS ARRAY_PARTITION variable=work->p    dim=1 complete

// Cache matrices — complete → full register file, 1-cycle read
#pragma HLS ARRAY_PARTITION variable=cache->Kinf    complete
#pragma HLS ARRAY_PARTITION variable=cache->Quu_inv complete
#pragma HLS ARRAY_PARTITION variable=cache->AmBKt   complete

// Dataflow for concurrent sub-step execution
#pragma HLS DATAFLOW
```

---

## Step 5 — Unconventional Resource Usage (UltraScale+ specific)

| Technique | Where applied | Why |
|-----------|---------------|-----|
| **CARRY8 comparators** | Box projection min/max | HLS default maps float cmp to LUT6 trees; CARRY8 is 2× faster, 0 DSPs used |
| **DSP58 pre-adder** | Dual update `g += x - vnew` | DSP58 `D - A + C` path absorbs subtraction free; saves 1 LUT per element |
| **SRL32 delay buffers** | Forward/backward pass pipeline inter-stage buffers | SRL32 uses 1 LUT per 32-bit shift reg vs 32 FFs; saves ~60% FF on delay lines |
| **URAM288** | Medium problems: Xref/Uref storage (NH>20) | URAM is 4× denser than BRAM36, avoids cascading; use `#pragma HLS BIND_STORAGE type=RAM_2P impl=URAM` |
| **LUT6 ROM** | SOC cone metadata (Acu, qcu — small int arrays) | Avoids wasting BRAM ports on tiny tables; inferred as distributed LUT ROM automatically |
| **DSP58 cascade** | Matrix-vector products for NX>8 | Chain DSP58s (`PCOUT → PCIN`) for long dot products; no routing between DSPs, saves 30% timing slack |

---

## Step 6 — Testbench (`testbench/tb_tinympc_hls.cpp`)

1. Instantiate solver with quadrotor problem (NX=12, NU=4, NH=10) — same params as
   `examples/problem_data/quadrotor_20hz_params.hpp`
2. Initialize `TinyCacheHLS` from pre-run software Riccati solve
3. Call `tinympc_solve<12,4,10>()` and compare `solution->u.col(0)` against software
   reference (tolerance 1e-4)
4. Report `iter` and `solved` flag

**Verification sequence:**
```bash
# C simulation (functional)
vitis_hls -f tcl/run_csim.tcl

# RTL cosim (timing + latency)
vitis_hls -f tcl/run_cosim.tcl

# Check synthesis report for II, latency, DSP/BRAM/FF/LUT counts
open build_hls/solution1/syn/report/tinympc_solve_csynth.rpt
```

---

## Step 7 — Tcl Project Script (`tcl/create_project.tcl`)

```tcl
open_project tinympc_hls_proj
set_top tinympc_solve
add_files src/tinympc_hls/admm_hls.cpp
add_files -tb src/tinympc_hls/testbench/tb_tinympc_hls.cpp
open_solution solution1 -flow_target vivado
set_part xczu9eg-ffvb1156-2-e   # ZU9EG as representative UltraScale+ part
create_clock -period 5 -name default  # 200 MHz
config_compile -name_max_length 80
csynth_design
cosim_design -trace_level all
export_design -format ip_catalog
```

---

## Implementation Order

1. **`types_hls.hpp`** — templated structs (no HLS dependencies, unit-testable in GCC)
2. **`admm_hls.cpp`** — implement `update_linear_cost` first (simplest, all parallel)
3. **`admm_hls.cpp`** — add `backward_pass_grad` + `forward_pass` (sequential, more complex)
4. **`admm_hls.cpp`** — add `update_slack` + `update_dual` + `check_termination`
5. **`tiny_api_hls.hpp`** — wire kernels into top-level with DATAFLOW + interface pragmas
6. **`testbench/tb_tinympc_hls.cpp`** — csim testbench
7. **`tcl/create_project.tcl`** — HLS project, run csim, check reports
8. Iterate on pragmas based on synthesis reports (target: II=1 on inner loops, <50K LUT)
