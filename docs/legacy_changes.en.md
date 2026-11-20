# Changes from the legacy code

[한국어](legacy_changes.md) | **English**

This page is for people who have used the 8 legacy scripts `EDRegisterSparse_Show_iter_*.m`. It summarizes what the new structure (`+edsm/`) keeps the same and what it changes, how each change was verified, and how to recover the legacy numbers. The legacy scripts have been deleted from the working tree and remain only in the git history.

## 1. Mapping from legacy scripts to new calls

Every new call first builds the model and the starting point.

```matlab
cfg   = edsm.defaultConfig();
src   = edsm.loadMesh('data/simple_poisson_mesh_020.ply');
tar   = edsm.loadMesh('data/simple_poisson_mesh_022.ply');
model = edsm.buildModel(src, tar, cfg);
P0    = edsm.initParam(model);
```

| Legacy script | New call | Notes |
|---|---|---|
| `..._gradient_hjh.m` | `edsm.solvers.gradientDescent(P0, model, struct('maxIter', 100, 'stepSize', 1e-3, 'globalStepScale', 0.01))` | The legacy script uses an older Jacobian variant (Section 3.D). Only the starting-point `err` can be reproduced numerically. |
| `..._Gauss_Newton_hjh_re.m` | `edsm.solvers.gaussNewton(P0, model, struct('maxIter', 100, 'stepTol', 1e-3/(1-1e-3)))` | `stepTol` is set to match the legacy stopping formula (Section 3.J). |
| `..._Gauss_Newton_delta_hjh.m` | `edsm.solvers.gaussNewton(P0, model, struct('jacobian', 'numeric', 'rankTol', 1e-5))` | Finite-difference Jacobian. The legacy script applied the term weights twice (Section 3.E). This is not reproduced. |
| `..._LM_hjh.m` | (not ported) | Replaced by `LM_hjh_2` (Section 3.F). |
| `..._LM_hjh_2.m` | `edsm.solvers.levenbergMarquardt(P0, model)` | The defaults are `damping 'diag'`, `linearSolver 'direct'`, `lambda0 0.01`. The legacy script had no stopping criterion, so to reproduce it pass `gradTol = 0, funTol = 0` (Section 4). |
| `..._CG_hjh.m` | `edsm.solvers.levenbergMarquardt(P0, model, struct('linearSolver', 'cg'))` | `damping` is `'diag'`. The legacy bug of frozen damping is fixed (Section 3.B). |
| `..._PCG_hjh.m` | `edsm.solvers.levenbergMarquardt(P0, model, struct('linearSolver', 'pcg'))` | The legacy `mode` (`'none'`, `'eye'`, `'diag'`) is the option `damping`. The preconditioner is Jacobi (`diag(A)`). |
| `..._Jall_FINAL.m` | `edsm.solvers.lsqnonlinBaseline(P0, model)` | Requires the Optimization Toolbox. It uses the same options as the legacy script (analytic Jacobian, `FunctionTolerance 1e-10`, `MaxIterations 30`). |

The option names and defaults are defined by the help header of each function (`help edsm.solvers.levenbergMarquardt`, etc.). Every solver has the form `[P, hist] = solver(P0, model, opts)`, and passing an unknown field in `opts` raises the error `edsm:solvers:unknownOption` (to catch typos).

The settings that the legacy scripts hard-coded match the defaults of `edsm.defaultConfig()` (`nVertexNeighbor 4`, `nNodeNeighbor 6`, `sampleRate 0.01`, `wRot 1.0`, `wSmooth 0.1`, `wLap 0.01`, `distanceOnMesh true`, `seed 1`). To change these, change the fields of `cfg` and pass it to `edsm.buildModel`.

## 2. Removed dependencies

| What the legacy code needed | New code |
|---|---|
| mexopencv (`cv.Rodrigues`) | `edsm.rodrigues` (pure MATLAB, includes the Jacobian) |
| Statistics Toolbox (`knnsearch`) | `edsm.nearestIdx` (exact brute-force search; ties go to the smallest row index) |
| GPU (`gpuArray`) | not used |
| private package `jwutils` (`plyread`, etc.) | `edsm.loadMesh`, `edsm.showMesh`, `edsm.showMeshes` |

The core of `+edsm` requires only base MATLAB R2020b or later. `lsqnonlinBaseline` requires the Optimization Toolbox, and the `parfor` in `numericJacobian` is optional. Development and testing were done only on R2026b (Apple Silicon), and earlier releases were not tested, so no compatibility is claimed. The percentile computation in the tutorial scripts also uses the helper `tutorials/helpers/pct.m` so that it works without the Statistics Toolbox.

## 3. Behavior differences and bugs fixed

### A. Three errors in the analytic Jacobian (fixed)

The legacy Jacobian had three errors. The default (`cfg.legacyJacobian = false`) is the corrected Jacobian, and `cfg.legacyJacobian = true` reproduces the legacy one bit for bit. $F$ is the same in both modes.

1. `Jrot1`/`Jrot2` were laid out for a row-major $A_i$, but $P$ stores $A(:)$ in column-major order. This is correct only when $A_i$ is symmetric (for example, at the identity starting point).
2. The rotation ($r$) columns of the Laplacian block were left at 0.
3. The 9 source vertices with no valid ED neighbors (small disconnected components with no nodes) are pinned to their source positions in $F$, but $J$ treated them as deformed.

**Impact.** With the legacy Jacobian, LM stops near `err` 162.4137 (`lambdaMax`, 33 iterations), and the gradient norm (2-norm) at that point is 42.46 when measured with the corrected Jacobian and 42.33 when measured with the legacy Jacobian. The corrected Jacobian reaches 150.6621 with a gradient norm of 0.2019 (Section 4 of [tutorial 06](06_levenberg_marquardt.en.md), `tutorials/output/t06_levenberg_marquardt.log`).

**Verification.**
- [Tutorial 03](03_jacobian.en.md) (`tutorials/t03_jacobian.m`) compares the analytic Jacobian against a finite difference (`edsm.numericJacobian`, correspondence fixed) block by block at a random point. For the corrected Jacobian, the maximum absolute error in every block is at or below 5.42e-09, and for the legacy Jacobian it is up to 9.64 (the fit-term $r$ columns) (`tutorials/output/t03_jacobian.log`).
- `tests/test_core.m` asserts the corrected Jacobian on all rows and columns, and only prints the legacy Jacobian. It also asserts that `F` does not depend on `legacyJacobian`.
- The bit-for-bit reproduction of the legacy mode is checked by `tests/test_regression.m` (Section 4 below).

### B. Damping frozen in CG/PCG (fixed)

The legacy CG/PCG scripts built the damped matrix $A=J^\top J+\lambda D$ **only once, before** the retry loop. After a step was rejected, increasing $\lambda$ by a factor of 10 did not change the system, so the same solution was rejected over and over, and for this reason the legacy scripts stopped after 4 iterations (CG) and 3 iterations (PCG). `levenbergMarquardt` rebuilds the matrix with the current $\lambda$ on every retry.

**Verification.** `tests/test_regression.m` asserts that the new solver, at the iteration after the one where the legacy script stopped (the 5th for CG, the 4th for PCG), goes through rejections and still continues (`h.nReject(nBL + 1) > 0`).

### C. The first iteration's trial-point computation deforms the invalid vertices (not reproduced)

In the legacy `LM_hjh_2`, `CG` and `PCG`, `EfuncLocal` overwrites `idxNeighbor(NaN)` in a local copy with 1 in the first outer iteration, and then calls itself to compute the trial point. As a result, the 9 invalid vertices are deformed in the first iteration's trial point (the first recorded error, the accept decision, and the previous error used by the 2nd iteration). The new solvers use the correct $F$ (invalid vertices stay at the source position). For this reason the legacy first-iteration error and the new solver's value differ slightly.

**Verification.** `tests/test_regression.m` evaluates $F$ at the first iterate with a model in which the neighbor indices of the invalid vertices are set to 1, and asserts that it matches the first error recorded by the legacy script (relative deviation `< 1e-12` for `lm_v2` and `< 1e-6` for `cg`/`pcg`).

### D. An older Jacobian variant in the gradient descent script (not reproduced)

The legacy `gradient_hjh` contains a Jacobian older than that of the other seven scripts (the rotation columns are not permuted, and the row indices of the $R$ block also differ). `legacyJacobian = true` reproduces the Jacobian of the seven scripts, so the iteration trajectory of the gradient descent script is not reproduced. Only the error before iterating (at the starting point) matches.

### E. The finite-difference GN script applies term weights twice (not reproduced)

The legacy `Gauss_Newton_delta_hjh` multiplied the finite-difference Jacobian by the weights twice. The new `numericJacobian` differences `edsm.residual` as it is, so the weights enter only once. The legacy numbers are not reproduced. To use a finite-difference Jacobian in the new code, also pass `rankTol` (about `1e-5`), which cuts noise-level singular values. This is because the exactly-zero singular values of a finite-difference Jacobian carry noise of about `1e-8` and are not cut by the default tolerance (see the `gaussNewton` help).

### F. Legacy LM v1 is not ported, `inv()` removed

The legacy `LM_hjh` (v1) was replaced by `LM_hjh_2` and was not ported. The `inv(...) * ...` used by the legacy LM family was changed to `A \ b`. Because of this, the error can differ at the relative `1e-6` level (the tolerance for the `lm_v2` comparison in `tests/test_regression.m` is `2e-6`).

### G. The `distanceOnMesh = false` branch (fixed)

In the legacy code, `Lap` was defined only inside the geodesic-distance branch, so the Euclidean-distance branch aborted with an error at the first place it used it. `edsm.buildModel` builds `Lap` with the same expression in both branches. `tests/test_core.m` exercises this branch.

### H. Rotation-vector convention

The legacy scripts' `encodeParam` used MATLAB's `rotationMatrixToVector`, whose convention is the transpose of `cv.Rodrigues` and so agrees only when $R=I$. The new `edsm.encodeParam` uses an inverse transform consistent with `edsm.rodrigues`. There is no difference at the identity starting point.

### I. New stopping criteria (behavior change)

The legacy LM had no convergence test. The new `levenbergMarquardt` uses by default `gradTol = 1e-6` (relative to the infinity norm of the gradient at the starting point) and `funTol = 1e-8` (small decrease 3 times in a row). So it can finish early even when you pass `maxIter = 100` (for example, the default setting ends at iteration 48 by `funTol`, tutorial 06). The previous behavior is `gradTol = 0, funTol = 0`.

### J. Interpretation of the Gauss–Newton stopping criterion

The legacy stopping formula compares its first two terms elementwise on the vector $F$, so it fires only when every component satisfies them. The third term is equivalent to $\|\delta\|_\infty<\epsilon_2/(1-\epsilon_2)=1.001001\times10^{-3}$. The new `gaussNewton` keeps only this third term and exposes it as `stepTol`. The default is `1e-3`, and to match the legacy exactly, pass `1e-3/(1-1e-3)`.

### K. Other

- `edsm.buildModel` saves and restores the global random state around node selection. The legacy scripts called `rng(1,'twister')` directly and so changed the global state.
- `edsm.loadMesh` returns the same $V$, $F$ as `jwutils.plyread` (`tests/test_core.m` (d) is skipped if the legacy package for comparison is not available).
- The `assert(~any(isnan(distNodeNeighbor(:))))` in `buildModel` does not catch unconnected nodes (distance `Inf`), so it is useless as a safeguard (the code carries the same comment).
- The definition of the objective $F$ itself (energy terms, normalization, weights) is the same. [Tutorial 02](02_energy.en.md) explains this definition.

## 4. Recovering the legacy numbers

1. **With the legacy Jacobian.** Build the model with `cfg = edsm.defaultConfig(); cfg.legacyJacobian = true;`.
2. **Turn off the stopping criteria.** For LM, pass `struct('gradTol', 0, 'funTol', 0)`. The legacy LM stopped only when `lambda > 1e5` (`lambdaMax`) or when it reached the iteration count (100). For GN, pass `stepTol = 1e-3/(1-1e-3)`.
3. **Run the regression test.** `tests/test_regression.m` runs the new solvers with the settings above and compares them with `tests/baseline/*.mat` (a record of the numbers from the legacy scripts; see `tests/baseline/README.md` for how it was made).

```matlab
setup
run('tests/test_regression.m')
```

The tolerances the test asserts are as follows (values from the code).

| Case | Assertion |
|---|---|
| `gradient` | The starting-point error matches to within `1e-14`. The rest differs because of Section 3.D and is only printed. |
| `gauss_newton` | The iteration count and stop reason (`stepTol`) are the same as the legacy, and the relative deviation over the first 18 iterations is below `1e-10`. After that the trajectory is a divergent, chaotic one and is not compared. |
| `lm_v2` | The $\lambda$ sequence (the used value and the next value) is exactly the same as the legacy and the stop reason is `lambdaMax`. From the 2nd iteration on, the relative deviation of the error is below `2e-6` and that of the final $P$ is below `5e-4`. The first iteration is explained separately because of Section 3.C. |
| `cg`, `pcg` | Up to where the legacy script stopped (4 iterations for CG, 3 for PCG), the error is within relative `1e-3`. The new solver goes through rejections at the next iteration and still continues. |
| `lsqnonlin` | The first 3 are below `1e-12` and the whole is below `1e-4`. The new call runs 31 iterations, and `hist.err`, including the starting point, has the same number of entries as the legacy record. |

Running it prints a table of the maximum relative deviation per case. This page does not quote those printed values directly; it quotes only the assertion tolerances from the code.

## 5. What is intentionally not reproduced

- The bit-for-bit reproduction of the legacy Jacobian is done with `legacyJacobian = true`, but the following are not reproduced.
  - The older Jacobian variant in the gradient descent script (Section 3.D)
  - The effect of invalid vertices being deformed in the first iteration's trial point (Section 3.C)
  - The early termination caused by the frozen damping in CG/PCG (Section 3.B)
  - The double application of weights in the finite-difference GN (Section 3.E)
  - LM v1 (`LM_hjh`) (Section 3.F)
- The legacy figure display, `pause`, and saving of results (`jwutils.saveMesh`, etc.) were not ported. The solvers do not draw figures; they report iteration information through `opts.callback` and `opts.verbose`. Visualization is handled by `edsm.showMesh`, `edsm.showMeshes` and the helpers in `tutorials/`.
