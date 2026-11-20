% TEST_REGRESSION  New solvers (+edsm/+solvers) vs the baselines of the legacy scripts.
%
%   Run:   matlab -batch "cd('<repo>'); setup; run('tests/test_regression.m')"
%
%   The model is built with cfg.legacyJacobian = true (bit-identical legacy
%   Jacobian, same F). Baselines: tests/baseline/*.mat (see its README).
%
%   Index mapping (hist.err(1) = error at P0, hist.err(k+1) = after iteration k):
%     gradient, gauss_newton : BL_ERR(k) was recorded BEFORE step k   -> hist.err(k)
%     lm_v2, cg, pcg         : BL_ERR(k) was recorded AFTER step k    -> hist.err(k+1)
%     lsqnonlin              : BL_ERR(1) is OutputFcn iteration 0     -> hist.err(k)
%
%   Known, explained deviations (derivations in the report / comments below):
%   [G]  gradient: the legacy gradient script contains an OLDER Jacobian variant
%        (J_R0 columns not permuted, different row index of the R block) than the
%        other seven scripts, which is what cfg.legacyJacobian reproduces. Only
%        the error at P0 (before any step) is comparable.
%   [Q]  lm_v2/cg/pcg: in the legacy EfuncLocal the first outer iteration
%        overwrites idxNeighbor(NaN) = 1 in its local copy BEFORE it calls itself
%        for the trial point. The trial F of iteration 1 (the recorded BL_ERR(1),
%        the accept test, and the F_old used by iteration 2) therefore deforms
%        the vertices without valid ED neighbors instead of keeping them at
%        their source position. The new solvers use the correct F. This is
%        checked below by evaluating F at the first iterate with a model whose
%        NaN neighbors are replaced by 1.
%   [GN] gauss_newton: the legacy iteration diverges (error up to 1e57), the
%        trajectory is chaotic; only the first iterations are compared tightly.
%   [C]  cg/pcg: the legacy scripts stop early (frozen damping, see report);
%        only the iterations before the first rejected step are comparable.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
bdir = fullfile(repoRoot, 'tests', 'baseline');
fprintf('==== regression of new solvers vs legacy baselines ====\n');

cfg = edsm.defaultConfig();
cfg.legacyJacobian = true;
src = edsm.loadMesh(fullfile(repoRoot, 'data', 'simple_poisson_mesh_020.ply'));
tar = edsm.loadMesh(fullfile(repoRoot, 'data', 'simple_poisson_mesh_022.ply'));
model = edsm.buildModel(src, tar, cfg);
P0 = edsm.initParam(model);

rows = {};     % name, nCompared, maxRel, argmax, dP, note
relDev = @(a, b) abs(a(:) - b(:)) ./ abs(b(:));
relP = @(P, bl) norm(P(:) - bl.BL_P0(:)) / norm(bl.BL_P0(:));

%% gradient -------------------------------------------------------------
bl = load(fullfile(bdir, 'gradient.mat'));
[P, h] = edsm.solvers.gradientDescent(P0, model, struct('maxIter', 100, 'stepSize', 1e-3, 'globalStepScale', 0.01));
assert(h.nIter == 100 && strcmp(h.stopReason, 'maxIter'));
r = relDev(h.err(1:100), bl.BL_ERR);
assert(r(1) < 1e-14, 'gradient: error at P0 differs from baseline');
rows(end+1, :) = {'gradient (all 100)', 100, max(r), find(r == max(r), 1), relP(P, bl), '[G] informational'};
fprintf('  gradient: rel dev of err by iteration 1..6: %s\n', sprintf('%.1e ', r(1:6)));
% Hold-out check of [G]: the gradient at P0 itself must agree with the legacy J
% only up to the older-variant difference, so the first iterate already differs.
assert(r(2) > 1e-4, 'gradient: expected the older-Jacobian deviation from iteration 2 on');

%% gauss_newton ---------------------------------------------------------
bl = load(fullfile(bdir, 'gauss_newton.mat'));
stepTolLegacy = 1e-3 / (1 - 1e-3);      % norm(d,inf) < 1e-3*(1+norm(d,inf))
[P, h] = edsm.solvers.gaussNewton(P0, model, struct('maxIter', 100, 'stepTol', stepTolLegacy));
nBL = numel(bl.BL_ERR);
assert(h.nIter == nBL && strcmp(h.stopReason, 'stepTol'), ...
    'gauss_newton: stopped after %d (%s), legacy %d', h.nIter, h.stopReason, nBL);
r = relDev(h.err(1:nBL), bl.BL_ERR);
KGN = 18;                                % chaotic beyond this, see rows below
assert(max(r(1:KGN)) < 1e-10, 'gauss_newton: first %d iterations deviate', KGN);
rows(end+1, :) = {sprintf('gauss_newton (1..%d)', KGN), KGN, max(r(1:KGN)), find(r(1:KGN) == max(r(1:KGN)), 1), NaN, 'asserted < 1e-10'};
rows(end+1, :) = {sprintf('gauss_newton (all %d)', nBL), nBL, max(r), find(r == max(r), 1), relP(P, bl), 'stop iter == legacy; [GN]'};
fprintf('  gauss_newton: stopped at iteration %d = legacy %d (stepTol); err(%d) legacy %.9g new %.9g\n', ...
    h.nIter, nBL, nBL, bl.BL_ERR(end), h.err(nBL));

%% lm_v2 ------------------------------------------------------------------
% (the legacy scripts have no convergence test: LM runs here pass gradTol = 0, funTol = 0;
%  with the legacy Jacobian the default funTol would stop several runs before lambdaMax)
modelQ = model;                          % [Q] reproduction of the legacy idxNeighbor overwrite
ix = model.idxNeighbor; ix(isnan(ix)) = 1; modelQ.idxNeighbor = ix;
fprintf('  vertices without valid ED neighbors: %d\n', nnz(any(isnan(model.idxNeighbor), 1)));

bl = load(fullfile(bdir, 'lm_v2.mat'));
[P, h] = edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 100, 'damping', 'diag', 'linearSolver', 'direct', 'gradTol', 0, 'funTol', 0));
nBL = numel(bl.BL_ERR);
assert(h.nIter == nBL && strcmp(h.stopReason, 'lambdaMax'), 'lm_v2: nIter %d stop %s', h.nIter, h.stopReason);
assert(isequal(h.lambda, bl.BL_LAM) && isequal(h.lambdaNext, bl.BL_LAMOUT), 'lm_v2: lambda sequence differs');
P1 = firstIterate(P0, model, 'direct');
q = relDev(sum(edsm.residual(P1, modelQ).^2), bl.BL_ERR(1));
assert(q < 1e-12, 'lm_v2: [Q] explanation not confirmed (%.2e)', q);
r = relDev(h.err(2:nBL+1), bl.BL_ERR);
assert(r(1) < 1e-3 && max(r(2:end)) < 2e-6, 'lm_v2: err deviation too large');
assert(relP(P, bl) < 5e-4, 'lm_v2: final P deviates');
rows(end+1, :) = {'lm_v2 iter 1', 1, r(1), 1, NaN, sprintf('[Q]; F(P1) with Q-model vs BL: %.1e', q)};
rows(end+1, :) = {sprintf('lm_v2 (2..%d)', nBL), nBL - 1, max(r(2:end)), 1 + find(r(2:end) == max(r(2:end)), 1), relP(P, bl), 'asserted < 2e-6; lambda seq identical; P: asserted < 5e-4'};
fprintf('  lm_v2: lambda sequence (used and next) identical for all %d iterations; stop = lambdaMax like legacy\n', nBL);

%% cg and pcg -------------------------------------------------------------
for variant = {'cg', 'pcg'}
    v = variant{1};
    bl = load(fullfile(bdir, [v '.mat']));
    nBL = numel(bl.BL_ERR);
    [P, h] = edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 100, 'damping', 'diag', 'linearSolver', v, 'gradTol', 0, 'funTol', 0));
    % The legacy run is comparable for iterations 1..nBL, which were all accepted at the
    % first trial; iteration nBL+1 is the one where the legacy retry loop with the frozen matrix fails.
    assert(isequal(h.lambda(1:nBL), bl.BL_LAM) && all(h.nReject(1:nBL) == 0), '%s: lambda/reject mismatch', v);
    assert(h.nReject(nBL + 1) > 0, '%s: expected a rejection at iteration %d', v, nBL + 1);
    P1 = firstIterate(P0, model, v);
    q = relDev(sum(edsm.residual(P1, modelQ).^2), bl.BL_ERR(1));
    assert(q < 1e-6, '%s: [Q] explanation not confirmed (%.2e)', v, q);
    r = relDev(h.err(2:nBL+1), bl.BL_ERR);
    assert(max(r) < 1e-3, '%s: err deviation too large', v);
    [Pn, ~] = edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', nBL, 'damping', 'diag', 'linearSolver', v, 'gradTol', 0, 'funTol', 0));
    rows(end+1, :) = {sprintf('%s (1..%d)', v, nBL), nBL, max(r), find(r == max(r), 1), relP(Pn, bl), ...
        sprintf('[Q]+[C]; F(P1) Q-model vs BL %.1e; new run rejects at it %d (%d times)', q, nBL + 1, h.nReject(nBL + 1))};
    fprintf('  %s: rel dev of err by iteration: %s\n', v, sprintf('%.1e ', r));
    fprintf('  %s: legacy stopped after %d iterations; new solver continues to %d iterations (stop %s)\n', v, nBL, h.nIter, h.stopReason);
end

%% lsqnonlin --------------------------------------------------------------
bl = load(fullfile(bdir, 'lsqnonlin.mat'));
[P, h] = edsm.solvers.lsqnonlinBaseline(P0, model);
nBL = numel(bl.BL_ERR);
assert(numel(h.err) == nBL, 'lsqnonlin: %d err entries vs %d', numel(h.err), nBL);
r = relDev(h.err, bl.BL_ERR);
assert(max(r(1:3)) < 1e-12 && max(r) < 1e-4, 'lsqnonlin: deviation too large');
rows(end+1, :) = {'lsqnonlin (1..3)', 3, max(r(1:3)), find(r(1:3) == max(r(1:3)), 1), NaN, 'asserted < 1e-12'};
rows(end+1, :) = {sprintf('lsqnonlin (all %d)', nBL), nBL, max(r), find(r == max(r), 1), relP(P, bl), 'asserted < 1e-4'};
fprintf('  lsqnonlin: %d iterations (legacy: %d OutputFcn entries incl. iteration 0), stop %s\n', h.nIter, nBL, h.stopReason);

%% table --------------------------------------------------------------------
fprintf('\n%-22s %9s %14s %8s %14s   %s\n', 'case', 'compared', 'max rel dev', 'at k', 'rel dev final P', 'note');
for i = 1:size(rows, 1)
    fprintf('%-22s %9d %14.3e %8d %14.3e   %s\n', rows{i, :});
end
fprintf('\nAll regression assertions passed.\n');

function P1 = firstIterate(P0, model, solver)
% parameters after the first accepted LM iteration
if strcmp(solver, 'direct')
    o = struct('maxIter', 1, 'damping', 'diag', 'linearSolver', 'direct');
else
    o = struct('maxIter', 1, 'damping', 'diag', 'linearSolver', solver);
end
P1 = edsm.solvers.levenbergMarquardt(P0, model, o);
end
