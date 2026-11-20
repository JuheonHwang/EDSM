%% t08_compare  All solvers from the same start
% Runs gradient descent, Gauss-Newton, three LM variants (direct/diag, direct/eye,
% CG and Jacobi-PCG as linear solver) and MATLAB's lsqnonlin on the same
% problem with the corrected Jacobian, and compares error versus iteration,
% error versus time, and the final numbers. The runs are cached in
% tutorials/output/t08_runs.mat; set forceRecompute = true to redo them.
clearvars; close all;
forceRecompute = false;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'helpers'));
ctx = tutorialInit('t08_compare');
col = tutorialColors();

%% 1. Problem and solver list
[model, P0] = loadTutorialModel(ctx);
names = {'gradient descent', 'Gauss-Newton', 'LM direct/diag', 'LM direct/eye', 'LM cg/diag', 'LM pcg/diag', 'lsqnonlin'};
colors = {col.gd, col.gn, col.lm, col.lmEye, col.lmCG, col.lmPCG, col.lsq};
runFun = {
    @() edsm.solvers.gradientDescent(P0, model, struct('maxIter', 100))
    @() edsm.solvers.gaussNewton(P0, model, struct('maxIter', 30))
    @() edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 100))
    @() edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 100, 'damping', 'eye'))
    @() edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 100, 'linearSolver', 'cg'))
    @() edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 100, 'linearSolver', 'pcg'))
    @() edsm.solvers.lsqnonlinBaseline(P0, model, struct('maxIter', 30))
    };
nS = numel(names);

%% 2. Run (or load) the solvers
% Each run is timed by the solver itself (hist.time). The cache stores P and
% hist of every solver.
cacheFile = fullfile(ctx.outDir, 't08_runs.mat');
useCache = ~forceRecompute && exist(cacheFile, 'file');
if useCache
    C = load(cacheFile);
    useCache = isequal(C.names, names);
end
if useCache
    fprintf('Loaded cached runs from tutorials/output/t08_runs.mat (set forceRecompute = true to redo)\n');
    Pall = C.Pall; Hall = C.Hall;
else
    Pall = cell(1, nS); Hall = cell(1, nS);
    for k = 1:nS
        ws = warning('off', 'all');
        [Pall{k}, Hall{k}] = runFun{k}();
        warning(ws);
        fprintf('ran %-18s: %3d iterations, stop "%s", %.1f s\n', names{k}, Hall{k}.nIter, Hall{k}.stopReason, Hall{k}.time(end));
    end
    save(cacheFile, 'names', 'Pall', 'Hall');
end

%% 3. Final table
% Gradient norm (2-norm) = ||2 J'F|| at the returned parameters, computed here for all
% solvers with the corrected Jacobian.
fin = zeros(nS, 1); mn = zeros(nS, 1); gnorm = zeros(nS, 1); tt = zeros(nS, 1);
fprintf('\n%-18s %5s %-12s %14s %14s %12s %9s\n', 'solver', 'iter', 'stop', 'final err', 'min err', '|grad|_2', 'time [s]');
for k = 1:nS
    [Fk, Jk] = edsm.residual(Pall{k}, model);
    fin(k) = sum(Fk.^2); mn(k) = min(Hall{k}.err); gnorm(k) = norm(2 * (Jk' * Fk)); tt(k) = Hall{k}.time(end);
    fprintf('%-18s %5d %-12s %14.6g %14.6g %12.4g %9.2f\n', names{k}, Hall{k}.nIter, Hall{k}.stopReason, fin(k), mn(k), gnorm(k), tt(k));
end
fprintf('(final err is recomputed from the returned P; for lsqnonlin / GD it can differ from hist.err(end) only by rounding)\n');
ok = isfinite(fin);
[bestErr, ib] = min(fin(ok)); okIdx = find(ok); ib = okIdx(ib);
fprintf('\nlowest final error: %s with %.4f (start %.4f)\n', names{ib}, bestErr, Hall{1}.err(1));
fprintf('final errors of the converging solvers (LM variants, lsqnonlin) span %.4f - %.4f; differences of this size between\n', min(fin(3:7)), max(fin(3:7)));
fprintf('runs that stopped for different reasons are not a ranking of the methods.\n');
for k = 1:nS
    if isfield(Hall{k}, 'gradNorm')
        fprintf('hist.gradNorm (inf-norm) for %-16s final %.4g\n', names{k}, Hall{k}.gradNorm(end));
    end
end

%% 4. Figure: error vs iteration and vs time
% Same y axis; y is clipped to [140, 1000], so gradient descent (which
% oscillates above the start value) and Gauss-Newton (which leaves the chart
% at iteration 2) are partly or fully outside.
fig = newFig(ctx, 8, 4.8);
ax1 = axes(fig, 'Position', [0.09 0.25 0.40 0.65]); hold(ax1, 'on');
ax2 = axes(fig, 'Position', [0.58 0.25 0.39 0.65]); hold(ax2, 'on');
hl = gobjects(1, nS);
for k = 1:nS
    h = Hall{k};
    hl(k) = plot(ax1, 0:numel(h.err) - 1, h.err, '-', 'Color', colors{k}, 'DisplayName', names{k});
    plot(ax2, max(h.time, 1e-2), h.err, '-', 'Color', colors{k});
end
for ax = [ax1 ax2]
    set(ax, 'YScale', 'log'); ylim(ax, [140 1000]); yticks(ax, [150 200 300 500 1000]);
end
set(ax1, 'XScale', 'linear'); xlim(ax1, [0 100]);
set(ax2, 'XScale', 'log'); xlim(ax2, [1e-2, 40]);
xlabel(ax1, 'iteration'); ylabel(ax1, 'error  \Sigma F_k^2'); title(ax1, 'Error vs iteration');
xlabel(ax2, 'time [s]'); ylabel(ax2, 'error  \Sigma F_k^2'); title(ax2, 'Error vs time');
lg = legend(ax1, hl, 'Orientation', 'horizontal', 'NumColumns', 4, 'FontSize', 8, 'Location', 'southoutside');
lg.Position = [0.05 0.0 0.92 0.09];
saveFig(ctx, fig, 'convergence');

% Zoom on the solvers that converge (y axis 149-156).
fig = newFig(ctx, 8, 4.8);
ax1 = axes(fig, 'Position', [0.09 0.25 0.40 0.65]); hold(ax1, 'on');
ax2 = axes(fig, 'Position', [0.58 0.25 0.39 0.65]); hold(ax2, 'on');
hl = gobjects(1, nS - 2);
for k = 3:nS
    h = Hall{k};
    hl(k - 2) = plot(ax1, 0:numel(h.err) - 1, h.err, '-', 'Color', colors{k}, 'DisplayName', names{k});
    plot(ax2, max(h.time, 1e-2), h.err, '-', 'Color', colors{k});
end
ylim(ax1, [149 156]); ylim(ax2, [149 156]); xlim(ax1, [0 100]); set(ax2, 'XScale', 'log'); xlim(ax2, [1e-1, 40]);
xlabel(ax1, 'iteration'); ylabel(ax1, 'error  \Sigma F_k^2'); title(ax1, 'Converging solvers, zoom');
xlabel(ax2, 'time [s]'); ylabel(ax2, 'error  \Sigma F_k^2'); title(ax2, 'Converging solvers, zoom');
lg = legend(ax1, hl, 'Orientation', 'horizontal', 'NumColumns', 3, 'FontSize', 8, 'Location', 'southoutside');
lg.Position = [0.05 0.0 0.92 0.09];
saveFig(ctx, fig, 'convergence_zoom');

%% 5. Figure: final error and run time
fig = newFig(ctx, 8, 4);
ax = axes(fig, 'Position', [0.09 0.30 0.39 0.62]);
b = bar(ax, categorical(names, names), min(fin, 318), 'FaceColor', 'flat');   % Gauss-Newton is clipped, see label
for k = 1:nS, b.CData(k, :) = colors{k}; end
ylabel(ax, 'final error'); title(ax, 'Final error (axis starts at 140)');
ax.XTickLabelRotation = 40; ylim(ax, [140 320]);
yline(ax, Hall{1}.err(1), ':', 'Color', 'k');
text(ax, 2, 315, sprintf('%.1e', fin(2)), 'HorizontalAlignment', 'center', 'FontSize', 9, 'VerticalAlignment', 'top');
text(ax, 2, 295, '(off scale)', 'HorizontalAlignment', 'center', 'FontSize', 8, 'VerticalAlignment', 'top');
ax = axes(fig, 'Position', [0.58 0.30 0.39 0.62]);
b = bar(ax, categorical(names, names), tt, 'FaceColor', 'flat');
for k = 1:nS, b.CData(k, :) = colors{k}; end
ylabel(ax, 'time [s]'); title(ax, 'Run time'); ax.XTickLabelRotation = 40; ylim(ax, [0 1.1 * max(tt)]);
saveFig(ctx, fig, 'summary');

%% 6. Render the best result against the target
% Before/after overlays (blue = source, orange = target) and the distance
% from every target vertex to the nearest deformed vertex.
Pb = Pall{ib};
Vb = edsm.deform(Pb, model);
d0 = fitDistance(model.src.V, model.tar.V);
d1 = fitDistance(Vb, model.tar.V);
fprintf('\nbest result (%s): target-to-source distance mean %.2f -> %.2f, 95%% %.2f -> %.2f, max %.2f -> %.2f\n', names{ib}, ...
    mean(d0), mean(d1), pct(d0, 95), pct(d1, 95), max(d0), max(d1));
bnd = meshBounds(model.src.V, model.tar.V, Vb);
cmax = 40;
fig = newFig(ctx, 8.6, 5.6);
axs = panelGrid(fig, 1, 4, 'Margins', [0.005 0.005 0.11 0.10], 'Gap', [0.005 0]);
drawOverlay(axs(1), model.src.V, model.src.F, model.tar.V, model.tar.F); setBodyView(axs(1), bnd); panelTitle(axs(1), 'before: overlay', 'FontSize', 10);
drawOverlay(axs(2), Vb, model.src.F, model.tar.V, model.tar.F); setBodyView(axs(2), bnd); panelTitle(axs(2), 'after: overlay', 'FontSize', 10);
dd = {d0, d1}; tl = {'before: fit error', 'after: fit error'};
for q = 1:2
    ax = axs(2 + q);
    drawSurface(ax, model.tar.V, model.tar.F, 'CData', min(dd{q}, cmax));
    ax.Colormap = viridisMap(256); ax.CLim = [0 cmax];
    setBodyView(ax, bnd); panelTitle(ax, tl{q}, 'FontSize', 10);
end
cb = colorbar(axs(4), 'Location', 'southoutside');
cb.Position = [0.55 0.065 0.40 0.022];
cb.Label.String = 'distance to nearest source vertex (mesh units)'; cb.FontSize = 9;
annotation(fig, 'textbox', [0.0 0.0 0.5 0.09], 'String', {sprintf('best: %s, err %.1f -> %.1f', names{ib}, Hall{1}.err(1), bestErr), 'blue = source, orange = target'}, ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 10);
saveFig(ctx, fig, 'best_result');

tutorialDone(ctx);
