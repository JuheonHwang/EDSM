# Documentation index

[한국어](README.md) | **English**

We recommend reading the pages in the order below. Every page has the same structure (what you will learn, how to run, body, results, summary) and explains the script of the same name (in `../tutorials/`) and its run log (in `../tutorials/output/`). Every number in the pages is taken from that log.

| Order | Page | Content |
|---|---|---|
| 1 | [00. Quickstart](00_quickstart.en.md) | Run the whole pipeline once. |
| 2 | [01. The ED model and its parameters](01_ed_model.en.md) | Nodes, weights, the deformation formula, and the layout of the parameter vector $P$. |
| 3 | [02. The objective and its energy terms](02_energy.en.md) | The four terms of the residual $F$, their normalization, the correspondences, and where the code differs from the textbook. |
| 4 | [03. The analytic Jacobian](03_jacobian.en.md) | Structure, finite-difference verification, errors in the legacy Jacobian, rank deficiency. |
| 5 | [04. Gradient descent](04_gradient_descent.en.md) | The stability limit and oscillation. |
| 6 | [05. Gauss–Newton](05_gauss_newton.en.md) | A good first step and then divergence; small singular values. |
| 7 | [06. Levenberg–Marquardt](06_levenberg_marquardt.en.md) | Damping, control of $\lambda$, and the effect of the legacy Jacobian. |
| 8 | [07. CG and PCG](07_cg_pcg.en.md) | The condition number of LM's linear system and iterative solvers. |
| 9 | [08. Comparing all solvers](08_compare.en.md) | Comparison in one table and how to read the numbers. |
| Appendix | [Differences from the original code](legacy_changes.en.md) | Correspondence to the 8 original scripts, bug fixes, and how to reproduce the original numbers. |

If you are new, run 00 first. Read 01 to 03 if you are curious about the problem setup, and 04 to 08 if your goal is to compare the optimizers. An introduction to the whole repository is in [../README.en.md](../README.en.md).

The figures are in `img/` (the file name is the script name joined with the figure name, for example `img/t00_quickstart_before_after.png`). Rerunning a script regenerates them under the same names.
