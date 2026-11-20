% TEST_SOLVERS  Unit checks of the +edsm/+solvers package.
%
%   Run from anywhere after setup:
%       matlab -batch "cd('<repo>'); setup; run('tests/test_solvers.m')"
%
%   (a) conjGrad (both preconditioners) vs backslash on a random SPD system
%   (b) every solver runs 3 iterations (corrected Jacobian) and returns a
%       well-formed hist
%   (c) gaussNewton with numeric vs analytic Jacobian: first step, corrected mode
%   (d) levenbergMarquardt convergence criteria (gradTol / funTol) and hist.gradNorm

repoRoot = fileparts(fileparts(mfilename('fullpath')));
fprintf('==== edsm solver unit tests ====\n');

%% (a) conjGrad
fprintf('\n[a] conjGrad vs backslash, random SPD (n = 300)\n');
rngBackup = rng; rng(7);
n = 300;
B = randn(n);
A = B' * B + 5 * eye(n);          % SPD, cond about 1e3
b = randn(n, 1);
xref = A \ b;
for pc = {'none', 'jacobi'}
    [x, res] = edsm.solvers.conjGrad(A, b, [], struct('precond', pc{1}));
    relErr = norm(x - xref) / norm(xref);
    fprintf('    precond %-6s : %3d iterations, final |r| = %.2e, rel. error vs A\\b = %.2e\n', ...
        pc{1}, numel(res), res(end), relErr);
    assert(relErr < 1e-8, 'conjGrad(%s) deviates from A\\b', pc{1});
    assert(res(end) < 1e-10 && numel(res) <= n);
end
% sparse input and nonzero x0 give the same answer
[x, ~] = edsm.solvers.conjGrad(sparse(A), b, randn(n, 1), struct('precond', 'jacobi'));
assert(norm(x - xref) / norm(xref) < 1e-8, 'sparse / x0 variant');
% already-converged start: zero iterations
[~, res0] = edsm.solvers.conjGrad(A, b, xref, struct('tol', 1e-6));
assert(isempty(res0));
rng(rngBackup);

%% model (corrected Jacobian, default config)
model = edsm.buildModel(edsm.loadMesh(fullfile(repoRoot, 'data', 'simple_poisson_mesh_020.ply')), ...
                        edsm.loadMesh(fullfile(repoRoot, 'data', 'simple_poisson_mesh_022.ply')));
assert(~model.cfg.legacyJacobian);
P0 = edsm.initParam(model);

%% (b) every solver, 3 iterations
fprintf('\n[b] 3 iterations of every solver (corrected Jacobian)\n');
fprintf('    %-26s %5s %-18s %12s %12s\n', 'solver', 'nIter', 'stopReason', 'err(1)', 'err(end)');
cases = {
    'gradientDescent',        @(o) edsm.solvers.gradientDescent(P0, model, o), struct('maxIter', 3)
    'gaussNewton',            @(o) edsm.solvers.gaussNewton(P0, model, o),     struct('maxIter', 3)
    'LM direct/diag',         @(o) edsm.solvers.levenbergMarquardt(P0, model, o), struct('maxIter', 3)
    'LM direct/eye',          @(o) edsm.solvers.levenbergMarquardt(P0, model, o), struct('maxIter', 3, 'damping', 'eye')
    'LM direct/none',         @(o) edsm.solvers.levenbergMarquardt(P0, model, o), struct('maxIter', 3, 'damping', 'none')
    'LM cg/diag',             @(o) edsm.solvers.levenbergMarquardt(P0, model, o), struct('maxIter', 3, 'linearSolver', 'cg')
    'LM pcg/diag',            @(o) edsm.solvers.levenbergMarquardt(P0, model, o), struct('maxIter', 3, 'linearSolver', 'pcg')
    'lsqnonlinBaseline',      @(o) edsm.solvers.lsqnonlinBaseline(P0, model, o),  struct('maxIter', 3)
    };
cbMap = containers.Map('KeyType', 'char', 'ValueType', 'double');   % handle object: counts callback calls
for c = 1:size(cases, 1)
    o = cases{c, 3};
    cbMap('n') = 0;
    o.callback = @(it, P, h) countCallback(cbMap, it, P, h);   % must be called once per iteration
    ws = warning('off', 'all');
    [P, h] = cases{c, 2}(o);
    warning(ws);
    checkHist(h, cases{c, 1});
    assert(numel(P) == numel(P0) && all(isfinite(P)), '%s: bad P', cases{c, 1});
    assert(cbMap('n') == h.nIter, '%s: callback called %d times for %d iterations', cases{c, 1}, cbMap('n'), h.nIter);
    assert(h.nIter >= 1 && h.nIter <= 3 + strcmp(cases{c, 1}, 'lsqnonlinBaseline'), '%s: nIter %d', cases{c, 1}, h.nIter);
    if ~strcmp(cases{c, 1}, 'gaussNewton')      % undamped GN is not monotone (diverges on this problem)
        assert(h.err(end) < h.err(1), '%s: no progress in 3 iterations', cases{c, 1});
    end
    fprintf('    %-26s %5d %-18s %12.6f %12.6f\n', cases{c, 1}, h.nIter, h.stopReason, h.err(1), h.err(end));
end
% unknown option is rejected
try
    edsm.solvers.gradientDescent(P0, model, struct('maxIters', 3));
    error('unknown option accepted');
catch ex
    assert(strcmp(ex.identifier, 'edsm:solvers:unknownOption'));
end

%% (c) GN numeric vs analytic
fprintf('\n[c] gaussNewton, first step, numeric vs analytic Jacobian (corrected mode)\n');
tA = tic; [Pa, ha] = edsm.solvers.gaussNewton(P0, model, struct('maxIter', 1)); tA = toc(tA);
da = Pa - P0;
% (c1) default lsqminnorm tolerance: the FD Jacobian has near-null singular values
%      ~1e-8 (analytic: ~1e-15) that are not truncated -> huge step in null directions
[Pn0, hn0] = edsm.solvers.gaussNewton(P0, model, struct('maxIter', 1, 'jacobian', 'numeric'));
d0 = Pn0 - P0;
fprintf('    analytic                : |step| %.6g, err after step %.10g (%.1fs)\n', norm(da), ha.err(2), tA);
fprintf('    numeric, default rankTol: |step| %.6g, err after step %.4g, |dn-da|/|da| = %.3e   (not asserted)\n', ...
    norm(d0), hn0.err(2), norm(d0 - da) / norm(da));
% (c2) with an absolute rank tolerance above the FD noise (sigma_max ~ 6e2, true smallest nonzero ~ 3e-3)
tN = tic; [Pn, hn] = edsm.solvers.gaussNewton(P0, model, struct('maxIter', 1, 'jacobian', 'numeric', 'rankTol', 1e-5)); tN = toc(tN);
dn = Pn - P0;
fprintf('    numeric, rankTol 1e-5   : |step| %.6g, err after step %.10g (%.1fs)\n', norm(dn), hn.err(2), tN);
fprintf('    |step_num - step_ana| = %.3e  (relative %.3e), max abs %.3e; err rel. diff %.2e\n', ...
    norm(dn - da), norm(dn - da) / norm(da), max(abs(dn - da)), abs(ha.err(2) - hn.err(2)) / ha.err(2));
assert(norm(dn - da) / norm(da) < 1e-4, 'numeric and analytic first GN steps differ');

%% (d) LM convergence criteria
fprintf('\n[d] levenbergMarquardt gradTol / funTol, hist.gradNorm (corrected Jacobian)\n');
% (d1) default gradTol, funTol off: stops by gradTol within the cap; the criterion holds exactly
[Pg, hg] = edsm.solvers.levenbergMarquardt(P0, model, struct('funTol', 0, 'damping', 'eye'));
fprintf('    gradTol only (eye)  : nIter %3d, %s, gradNorm %.3e -> %.3e\n', hg.nIter, hg.stopReason, hg.gradNorm(1), hg.gradNorm(end));
assert(strcmp(hg.stopReason, 'gradTol') && hg.nIter < 100, 'gradTol did not stop the run');
assert(hg.gradNorm(end) <= 1e-6 * hg.gradNorm(1) && all(hg.gradNorm(1:end-1) > 1e-6 * hg.gradNorm(1)), 'gradTol criterion not met exactly at the last iterate');
checkHist(hg, 'LM gradTol');
% (d2) default funTol, gradTol off: stops by funTol after 3 consecutive small steps
[~, hf] = edsm.solvers.levenbergMarquardt(P0, model, struct('gradTol', 0, 'damping', 'eye'));
rd = -diff(hf.err) ./ (1 + hf.err(2:end));
fprintf('    funTol only (eye)   : nIter %3d, %s, last 3 rel. decreases %s\n', hf.nIter, hf.stopReason, sprintf('%.1e ', rd(end-2:end)));
assert(strcmp(hf.stopReason, 'funTol') && hf.nIter < 100 && all(rd(end-2:end) < 1e-8) && rd(end-3) >= 1e-8, 'funTol criterion');
% (d3) both disabled reproduces the pre-criteria run (LM direct/diag, corrected, stored in solver_comparison.mat)
old = load(fullfile(repoRoot, 'tests', 'results', 'solver_comparison.mat'), 'results');
ir = find(strcmp({old.results.mode}, 'corrected') & strcmp({old.results.solver}, 'LM direct/diag, no criteria'));
assert(isscalar(ir));
[~, h0] = edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 100, 'damping', 'diag', 'gradTol', 0, 'funTol', 0));
ho = old.results(ir).hist;
% Jacobian/factorization rounding differs slightly between machines/BLAS threads, so compare the
% error history with a loose relative tolerance (5e-6; the accepted-step sequence is the quantity that matters)
nh = min(numel(h0.err), numel(ho.err));
relDevErr = max(abs(h0.err(1:nh) - ho.err(1:nh)) ./ ho.err(1:nh));
fprintf('    both off, direct/diag: nIter %d (stored %d), %s, final err %.9f (stored %.9f), max rel dev %.2e\n', ...
    h0.nIter, ho.nIter, h0.stopReason, h0.err(end), ho.err(end), relDevErr);
assert(h0.nIter == ho.nIter && strcmp(h0.stopReason, ho.stopReason), 'gradTol=funTol=0 changed nIter/stopReason');
assert(relDevErr < 5e-6, 'gradTol=funTol=0 changed the error history');
% (d4) gradNorm well-formed; consistent with an independent evaluation at the final point
for hh = {hg, hf, h0}
    q = hh{1};
    assert(isequal(size(q.gradNorm), [q.nIter + 1, 1]) && all(isfinite(q.gradNorm)) && all(q.gradNorm >= 0), 'gradNorm shape/values');
end
[Fe, Je] = edsm.residual(Pg, model);
assert(abs(norm(2 * (Je' * Fe), inf) - hg.gradNorm(end)) <= 1e-9 * hg.gradNorm(1) + 1e-12, 'gradNorm(end) differs from 2*J''*F at the returned P');
% (d5) maxIter 0 still yields gradNorm(1); a threshold >= 1 (relative) stops at P0 with nIter 0
[Ps, hs] = edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 0));
assert(hs.nIter == 0 && isscalar(hs.gradNorm) && hs.gradNorm == hg.gradNorm(1) && isequal(Ps, P0(:)), 'maxIter 0');
[~, hc] = edsm.solvers.levenbergMarquardt(P0, model, struct('gradTol', 1, 'funTol', 0));
fprintf('    gradTol 1 (met at P0): nIter %d, %s\n', hc.nIter, hc.stopReason);
assert(hc.nIter == 0 && strcmp(hc.stopReason, 'gradTol'), 'gradTol at P0');

fprintf('\nAll solver unit tests passed.\n');

function countCallback(m, it, P, h)
m('n') = m('n') + 1;
assert(it == m('n') && it == h.nIter && ~isempty(P));
end

function checkHist(h, name)
need = {'err', 'stepNorm', 'time', 'nIter', 'stopReason'};
for i = 1:numel(need)
    assert(isfield(h, need{i}), '%s: hist.%s missing', name, need{i});
end
k = h.nIter;
assert(isequal(size(h.err), [k + 1, 1]), '%s: err size', name);
assert(isequal(size(h.stepNorm), [k, 1]), '%s: stepNorm size', name);
assert(isequal(size(h.time), [k + 1, 1]) && all(diff(h.time) >= 0), '%s: time', name);
assert(all(isfinite(h.err)) && all(h.err >= 0), '%s: err not finite/nonneg', name);
assert(ischar(h.stopReason) && ~isempty(h.stopReason), '%s: stopReason', name);
if isfield(h, 'lambda')
    assert(numel(h.lambda) == k && numel(h.lambdaNext) == k && numel(h.nReject) == k, '%s: lambda fields', name);
end
end
