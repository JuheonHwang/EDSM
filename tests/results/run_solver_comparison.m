% RUN_SOLVER_COMPARISON  Run every solver from edsm.initParam in both Jacobian modes.
%
%   matlab -batch "cd('<repo>'); setup; run('tests/results/run_solver_comparison.m')"
%
%   Default config; legacyJacobian = true and false. Solvers: gradientDescent (100),
%   gaussNewton (analytic, 30), levenbergMarquardt direct with damping diag/eye/none,
%   cg/diag, pcg/diag (maxIter 100 each), lsqnonlinBaseline (30). The LM runs use the
%   default convergence criteria (gradTol, funTol), except 'LM direct/diag, no criteria'
%   (gradTol = funTol = 0: the pre-criteria behavior, reference for tests/test_solvers.m).
%   Writes tests/results/solver_comparison.mat with variables
%     results : struct array (mode, solver, nIter, stopReason, errFirst, errMin,
%               errFinal, time, avgCgIters, gradNormFinal, P, hist)
%     summary : table of the scalar columns

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
src = edsm.loadMesh(fullfile(repoRoot, 'data', 'simple_poisson_mesh_020.ply'));
tar = edsm.loadMesh(fullfile(repoRoot, 'data', 'simple_poisson_mesh_022.ply'));

cases = {
    'gradientDescent',  @(P0, m) edsm.solvers.gradientDescent(P0, m, struct('maxIter', 100))
    'gaussNewton',      @(P0, m) edsm.solvers.gaussNewton(P0, m, struct('maxIter', 30))
    'LM direct/diag',   @(P0, m) edsm.solvers.levenbergMarquardt(P0, m, struct('maxIter', 100, 'damping', 'diag'))
    'LM direct/diag, no criteria', @(P0, m) edsm.solvers.levenbergMarquardt(P0, m, struct('maxIter', 100, 'damping', 'diag', 'gradTol', 0, 'funTol', 0))
    'LM direct/eye',    @(P0, m) edsm.solvers.levenbergMarquardt(P0, m, struct('maxIter', 100, 'damping', 'eye'))
    'LM direct/none',   @(P0, m) edsm.solvers.levenbergMarquardt(P0, m, struct('maxIter', 100, 'damping', 'none'))
    'LM cg/diag',       @(P0, m) edsm.solvers.levenbergMarquardt(P0, m, struct('maxIter', 100, 'damping', 'diag', 'linearSolver', 'cg'))
    'LM pcg/diag',      @(P0, m) edsm.solvers.levenbergMarquardt(P0, m, struct('maxIter', 100, 'damping', 'diag', 'linearSolver', 'pcg'))
    'lsqnonlin',        @(P0, m) edsm.solvers.lsqnonlinBaseline(P0, m, struct('maxIter', 30))
    };

results = struct('mode', {}, 'solver', {}, 'nIter', {}, 'stopReason', {}, 'errFirst', {}, 'errMin', {}, ...
    'errFinal', {}, 'time', {}, 'avgCgIters', {}, 'gradNormFinal', {}, 'P', {}, 'hist', {});
for legacy = [true false]
    cfg = edsm.defaultConfig();
    cfg.legacyJacobian = legacy;
    model = edsm.buildModel(src, tar, cfg);
    P0 = edsm.initParam(model);
    modeName = 'corrected'; if legacy, modeName = 'legacy'; end
    for c = 1:size(cases, 1)
        ws = warning('off', 'all');
        [P, h] = cases{c, 2}(P0, model);
        warning(ws);
        [F, J] = edsm.residual(P, model);
        r = struct('mode', modeName, 'solver', cases{c, 1}, 'nIter', h.nIter, 'stopReason', h.stopReason, ...
            'errFirst', h.err(1), 'errMin', min(h.err), 'errFinal', sum(F.^2), 'time', h.time(end), ...
            'avgCgIters', NaN, 'gradNormFinal', norm(2 * (J' * F)), 'P', P, 'hist', h);
        if isfield(h, 'linIters') && ~isempty(h.linIters), r.avgCgIters = mean(h.linIters); end
        results(end+1) = r; %#ok<SAGROW>
        fprintf('%-9s %-28s nIter %3d  %-18s err %10.4f -> min %10.4f final %10.4f  %6.1fs  cg %6.1f  |grad| %.3e\n', ...
            modeName, cases{c, 1}, r.nIter, r.stopReason, r.errFirst, r.errMin, r.errFinal, r.time, r.avgCgIters, r.gradNormFinal);
    end
end
summary = struct2table(rmfield(results, {'P', 'hist'}));
save(fullfile(repoRoot, 'tests', 'results', 'solver_comparison.mat'), 'results', 'summary', '-v7.3');
disp(summary);
