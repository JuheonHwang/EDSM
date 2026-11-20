%% t07_cg_pcg  Solving the LM linear system with CG and Jacobi-preconditioned CG
% Each LM iteration solves A d = b with A = J'J + lambda*diag(J'J), b = -J'F
% (606 x 606, symmetric positive definite for lambda > 0). 'direct' uses
% backslash; 'cg' and 'pcg' use conjugate gradients (edsm.solvers.conjGrad)
% without / with a Jacobi (diagonal) preconditioner. Part 1 takes one real
% system out of an LM run and compares the solvers on it; part 2 runs the
% complete LM with every linear solver.
clearvars; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'helpers'));
ctx = tutorialInit('t07_cg_pcg');
col = tutorialColors();

%% 1. One linear system taken from an actual LM iteration
% Run 5 LM iterations and build the system the solver would face next, with
% the lambda it would use (lambdaNext of iteration 5).
[model, P0] = loadTutorialModel(ctx);
[P5, h5] = edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 5));
lam = h5.lambdaNext(end);
[F, J] = edsm.residual(P5, model);
JtJ = J' * J; b = -(J' * F); n = numel(b);
A = JtJ + lam * spdiags(diag(JtJ), 0, n, n);
fprintf('System from LM iteration %d (err %.4f): lambda = %.1e, n = %d, nnz(A) = %d (%.1f%% dense), |b| = %.4g\n', ...
    h5.nIter, h5.err(end), lam, n, nnz(A), 100 * nnz(A) / n^2, norm(b));

tic; xd = A \ b; tDirect = toc;
fprintf('direct A\\b : %.4f s, residual |b - A x| = %.3e\n', tDirect, norm(b - A * xd));
solvers = {'cg', 'none'; 'pcg', 'jacobi'};
xs = cell(1, 2); rh = cell(1, 2);
for q = 1:2
    tic;
    [xs{q}, rh{q}] = edsm.solvers.conjGrad(A, b, zeros(n, 1), struct('precond', solvers{q, 2}));
    tt = toc;
    fprintf('%-4s (precond %-6s): %4d iterations, %.4f s, final residual %.3e, relative error vs direct %.3e\n', ...
        solvers{q, 1}, solvers{q, 2}, numel(rh{q}), tt, rh{q}(end), norm(xs{q} - xd) / norm(xd));
end
% The default cap is n = 606 iterations. How many would be needed?
capBig = 20000;
xbig = cell(1, 2); rbig = cell(1, 2);
for q = 1:2
    [xbig{q}, rbig{q}] = edsm.solvers.conjGrad(A, b, zeros(n, 1), struct('precond', solvers{q, 2}, 'maxIter', capBig));
    fprintf('%-4s with cap %d: %d iterations, final residual %.3e, relative error vs direct %.3e\n', solvers{q, 1}, capBig, numel(rbig{q}), rbig{q}(end), norm(xbig{q} - xd) / norm(xd));
end
for tol = [1e-10 1e-6 1e-3 1e-1 1]
    ic = find(rbig{1} < tol, 1); ip = find(rbig{2} < tol, 1);
    fprintf('first iteration with residual < %-6g : cg %s, pcg %s\n', tol, iterStr(ic), iterStr(ip));
end

fig = newFig(ctx, 8, 4.2);
ax = axes(fig, 'Position', [0.09 0.15 0.88 0.78]);
semilogy(ax, rbig{1}, '-', 'Color', col.lmCG, 'DisplayName', sprintf('CG (%d iterations to residual %.1e)', numel(rbig{1}), rbig{1}(end))); hold(ax, 'on');
semilogy(ax, rbig{2}, '-', 'Color', col.lmPCG, 'DisplayName', sprintf('Jacobi-PCG (%d iterations to residual %.1e)', numel(rbig{2}), rbig{2}(end)));
yline(ax, 1e-10, ':', 'Color', 'k', 'DisplayName', 'default tolerance 1e-10 (absolute)');
yline(ax, norm(b), '--', 'Color', [0.5 0.5 0.5], 'DisplayName', '|b| (initial residual)');
xline(ax, n, ':', 'Color', [0.5 0.5 0.5], 'DisplayName', 'default iteration cap = n = 606');
xlabel(ax, 'CG iteration'); ylabel(ax, '||b - A x||_2');
title(ax, 'Residual of one LM linear system'); legend(ax, 'Location', 'northeast', 'FontSize', 9);
xlim(ax, [0 2000]);   % CG itself ran up to 20000 iterations (see the log)
saveFig(ctx, fig, 'residual_curves');

%% 2. Why: spectrum and diagonal spread
% CG converges in few iterations when the eigenvalues are clustered; the
% number of iterations grows with the condition number. Jacobi-PCG applies
% CG to D^(-1/2) A D^(-1/2), D = diag(A), which has unit diagonal.
Af = full(A);
dA = diag(Af);
ev = sort(eig((Af + Af') / 2), 'descend');
Dh = 1 ./ sqrt(dA);
As = Dh .* Af .* Dh';
evs = sort(eig((As + As') / 2), 'descend');
fprintf('\ndiag(A): min %.3e, max %.3e, spread max/min = %.3e\n', min(dA), max(dA), max(dA) / min(dA));
fprintf('eigenvalues of A       : max %.4e, min %.4e, cond = %.3e\n', ev(1), ev(end), ev(1) / ev(end));
fprintf('eigenvalues of D^-1/2AD^-1/2: max %.4e, min %.4e, cond = %.3e\n', evs(1), evs(end), evs(1) / evs(end));
fprintf('smallest eigenvalue of the scaled matrix %.4e vs lambda %.1e (null directions of J''J are only lifted by the damping)\n', evs(end), lam);
fprintf('eigenvalues below 1e-3 * largest: A %d, scaled %d\n', sum(ev < 1e-3 * ev(1)), sum(evs < 1e-3 * evs(1)));
fprintf('eigenvalues below 1e-6 * largest: A %d, scaled %d\n', sum(ev < 1e-6 * ev(1)), sum(evs < 1e-6 * evs(1)));

fig = newFig(ctx, 8, 3.9);
ax = axes(fig, 'Position', [0.09 0.17 0.40 0.74]);
semilogy(ax, ev, '-', 'Color', col.lmCG, 'DisplayName', sprintf('A  (cond %.1e)', ev(1) / ev(end))); hold(ax, 'on');
semilogy(ax, evs, '-', 'Color', col.lmPCG, 'DisplayName', sprintf('D^{-1/2} A D^{-1/2}  (cond %.1e)', evs(1) / evs(end)));
xlabel(ax, 'index (sorted)'); ylabel(ax, 'eigenvalue'); title(ax, 'Eigenvalues'); legend(ax, 'Location', 'southwest', 'FontSize', 9);
ax = axes(fig, 'Position', [0.58 0.17 0.39 0.74]);
semilogy(ax, sort(dA, 'descend'), '-', 'Color', col.lm);
xlabel(ax, 'index (sorted)'); ylabel(ax, 'diag(A)'); title(ax, sprintf('Diagonal spread %.1e', max(dA) / min(dA)));
saveFig(ctx, fig, 'spectrum');

%% 3. Dependence on lambda
% LM shrinks lambda as it converges, which makes A more ill-conditioned:
% iteration counts of CG at fixed A for lambda over many decades (diag damping).
lams = 10.^(-8:1:0);
itCG = zeros(size(lams)); itPCG = zeros(size(lams)); cA = zeros(size(lams)); cS = zeros(size(lams));
for i = 1:numel(lams)
    Al = JtJ + lams(i) * spdiags(diag(JtJ), 0, n, n);
    [~, r1] = edsm.solvers.conjGrad(Al, b, zeros(n, 1), struct('precond', 'none'));
    [~, r2] = edsm.solvers.conjGrad(Al, b, zeros(n, 1), struct('precond', 'jacobi'));
    itCG(i) = numel(r1); itPCG(i) = numel(r2);
    Alf = full(Al); e1 = eig((Alf + Alf') / 2); dd = 1 ./ sqrt(diag(Alf)); Sf = dd .* Alf .* dd'; e2 = eig((Sf + Sf') / 2);
    cA(i) = max(e1) / min(e1); cS(i) = max(e2) / min(e2);
end
fprintf('\nlambda      cond(A)     cond(scaled)   CG iters   PCG iters   (cap %d)\n', n);
for i = 1:numel(lams)
    fprintf('%8.0e  %10.3e  %12.3e  %8d  %9d\n', lams(i), cA(i), cS(i), itCG(i), itPCG(i));
end
fig = newFig(ctx, 8, 3.9);
ax = axes(fig, 'Position', [0.09 0.17 0.40 0.74]);
semilogx(ax, lams, itCG, '-o', 'Color', col.lmCG, 'MarkerFaceColor', col.lmCG, 'DisplayName', 'CG'); hold(ax, 'on');
semilogx(ax, lams, itPCG, '-o', 'Color', col.lmPCG, 'MarkerFaceColor', col.lmPCG, 'DisplayName', 'Jacobi-PCG');
yline(ax, n, ':', 'Color', 'k', 'DisplayName', 'iteration cap');
xlabel(ax, '\lambda (diag damping)'); ylabel(ax, 'iterations to residual < 10^{-10}'); title(ax, 'Iterations vs \lambda');
legend(ax, 'Location', 'southwest', 'FontSize', 9); ylim(ax, [0 n * 1.1]);
ax = axes(fig, 'Position', [0.58 0.17 0.39 0.74]);
loglog(ax, lams, cA, '-o', 'Color', col.lmCG, 'MarkerFaceColor', col.lmCG, 'DisplayName', 'cond(A)'); hold(ax, 'on');
loglog(ax, lams, cS, '-o', 'Color', col.lmPCG, 'MarkerFaceColor', col.lmPCG, 'DisplayName', 'cond(D^{-1/2}AD^{-1/2})');
xlabel(ax, '\lambda (diag damping)'); ylabel(ax, 'condition number'); title(ax, 'Conditioning'); legend(ax, 'Location', 'northeast', 'FontSize', 9);
saveFig(ctx, fig, 'lambda_dependence');

%% 4. Full LM runs with each linear solver
% Same LM settings (diag damping, maxIter 100, default stopping criteria);
% only opts.linearSolver / opts.cgTol change. cgTol is an ABSOLUTE tolerance
% on the CG residual of every solve. The 'loose' variants use cgTol = 0.1,
% an inexact LM: each step is only an approximate solution of the system.
% (A tolerance of 1e-3 would not help plain CG: part 1 shows it needs
% thousands of iterations to get below that, so it still hits the cap.)
runs = {
    'direct',            struct('linearSolver', 'direct'),                        col.lm,    '-'
    'cg (tol 1e-10)',    struct('linearSolver', 'cg'),                            col.lmCG,  '-'
    'pcg (tol 1e-10)',   struct('linearSolver', 'pcg'),                           col.lmPCG, '-'
    'cg (tol 0.1)',      struct('linearSolver', 'cg', 'cgTol', 0.1),              col.lmCG,  '--'
    'pcg (tol 0.1)',     struct('linearSolver', 'pcg', 'cgTol', 0.1),             col.lmPCG, '--'
    };
nR = size(runs, 1);
H = cell(nR, 1); Pf = cell(nR, 1);
for k = 1:nR
    o = runs{k, 2}; o.maxIter = 100;
    [Pf{k}, H{k}] = edsm.solvers.levenbergMarquardt(P0, model, o);
end
fprintf('\n%-16s %5s %-10s %10s %12s %9s %12s %10s %10s\n', 'solver', 'iter', 'stop', 'final err', '|grad|_2', 'time [s]', 'solves', 'mean CG it', 'max CG it');
for k = 1:nR
    h = H{k};
    [Fk, Jk] = edsm.residual(Pf{k}, model);
    gnorm = norm(2 * (Jk' * Fk));
    if isempty(h.linIters)
        sMean = '-'; sMax = '-'; nSolve = numel(h.nReject) + sum(h.nReject);   % one solve per trial step
    else
        sMean = sprintf('%.1f', mean(h.linIters)); sMax = sprintf('%d', max(h.linIters)); nSolve = numel(h.linIters);
    end
    fprintf('%-16s %5d %-10s %10.4f %12.4g %9.2f %12d %10s %10s\n', runs{k, 1}, h.nIter, h.stopReason, h.err(end), gnorm, h.time(end), nSolve, sMean, sMax);
end
fprintf('(solves = linear solves incl. rejected trials; mean/max CG iterations per solve; - = direct)\n');
fprintf('hist.gradNorm available: %d\n', isfield(H{1}, 'gradNorm'));

fig = newFig(ctx, 8, 3.9);
ax = axes(fig, 'Position', [0.09 0.17 0.40 0.74]); hold(ax, 'on');
for k = 1:nR
    plot(ax, 0:H{k}.nIter, H{k}.err, runs{k, 4}, 'Color', runs{k, 3}, 'DisplayName', runs{k, 1});
end
set(ax, 'YScale', 'log'); ylim(ax, [145 300]);
xlabel(ax, 'accepted LM iteration'); ylabel(ax, 'error  \Sigma F_k^2'); title(ax, 'Error vs iteration');
legend(ax, 'Location', 'northeast', 'FontSize', 8);
ax = axes(fig, 'Position', [0.58 0.17 0.39 0.74]); hold(ax, 'on');
for k = 1:nR
    plot(ax, H{k}.time, H{k}.err, runs{k, 4}, 'Color', runs{k, 3});
end
set(ax, 'YScale', 'log', 'XScale', 'log'); ylim(ax, [145 300]);
xlabel(ax, 'time [s]'); ylabel(ax, 'error  \Sigma F_k^2'); title(ax, 'Error vs time');
saveFig(ctx, fig, 'lm_runs');

fig = newFig(ctx, 8, 3.9);
ax = axes(fig, 'Position', [0.09 0.17 0.88 0.74]); hold(ax, 'on');
for k = 2:nR
    li = H{k}.linIters;
    plot(ax, 1:numel(li), li, runs{k, 4}, 'Color', runs{k, 3}, 'DisplayName', runs{k, 1});
end
yline(ax, n, ':', 'Color', 'k', 'DisplayName', 'iteration cap 606');
set(ax, 'YScale', 'log');
xlabel(ax, 'linear solve number (all trial steps, in order)'); ylabel(ax, 'CG iterations');
title(ax, 'CG iterations per linear solve'); legend(ax, 'Location', 'southeast', 'FontSize', 9);
saveFig(ctx, fig, 'cg_iterations');

tutorialDone(ctx);

%% Local function
function s = iterStr(i)
if isempty(i), s = 'never'; else, s = sprintf('%d', i); end
end
