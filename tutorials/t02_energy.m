%% t02_energy  The residual vector and its four energy terms
% The objective is err(P) = sum(F(P).^2). F stacks four groups of residuals:
%   Efit    : target vertex minus its nearest deformed-source vertex    (3 per target vertex)
%   Erot    : A_i'A_i - I and det(A_i) - 1, "keep A_i a rotation"       (10 per node)
%   Esmooth : neighboring nodes must move consistently                  (3 per node)
%   Elap    : Laplacian of the deformed mesh, keeps surface detail      (3 per source vertex)
% Each is scaled and weighted (wRot = 1, wSmooth = 0.1, wLap = 0.01) as in
% the original scripts. This tutorial looks at their size at the start and
% after optimization, and at how each term reacts to specific changes of P.
clearvars; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'helpers'));
ctx = tutorialInit('t02_energy');
col = tutorialColors();

%% 1. Evaluate the residual at the starting point
% info.rows gives the row range of every term inside F (and inside the
% Jacobian). termEnergies sums the squares of F over those rows.
[model, P0] = loadTutorialModel(ctx);
n = model.nNodes;
[F0, ~, info0] = edsm.residual(P0, model);
r = info0.rows;
fprintf('\nRow ranges of F (numel(F) = %d):\n', numel(F0));
for f = fieldnames(r)'
    fprintf('  %-7s rows %5d : %5d   (%5d entries)\n', f{1}, r.(f{1})(1), r.(f{1})(2), diff(r.(f{1})) + 1);
end
[names, e0, rows] = termEnergies(F0, info0);
fprintf('\nContribution to err(P0) = sum(F.^2) = %.6f\n', sum(F0.^2));
fprintf('  %-8s %14s %10s\n', 'term', 'sum of squares', 'share');
for k = 1:4
    fprintf('  %-8s %14.6g %9.4f%%\n', names{k}, e0(k), 100 * e0(k) / sum(e0));
end
fprintf('  sum of the four terms - sum(F.^2) = %.2e\n', sum(e0) - sum(F0.^2));

%% 2. Why Elap is not zero at the start
% At P0 the mesh is undeformed, so Erot and Esmooth vanish. Elap does not: it
% is the Laplacian of the source itself (its local roughness), so it starts
% at a nonzero value and only penalizes deviation from that shape in the
% sense of keeping the Laplacian small (compare its value after LM below).
lapRes = reshape(F0(rows(4, 1):rows(4, 2)), [], 3);
fprintf('\nmax |Erot| = %.2e, max |Esmooth| = %.2e at P0\n', max(abs(F0(rows(2, 1):rows(2, 2)))), max(abs(F0(rows(3, 1):rows(3, 2)))));
fprintf('Elap: %d vertex residuals, mean |.| = %.4g, max |.| = %.4g (weight wLap = %.2f included)\n', size(lapRes, 1), ...
    mean(sqrt(sum(lapRes.^2, 2))), max(sqrt(sum(lapRes.^2, 2))), model.cfg.wLap);

%% 3. Correspondences of the fit term
% For every TARGET vertex the nearest vertex of the deformed source is found
% (info.indices), and the residual is their difference / sqrt(nTar). The
% correspondences are recomputed at every evaluation of F, as in ICP.
idx0 = info0.indices;
nTar = size(model.tar.V, 1);
fprintf('\ntarget vertices %d, distinct source vertices matched %d of %d (%.1f%%)\n', nTar, numel(unique(idx0)), model.nVerts, ...
    100 * numel(unique(idx0)) / model.nVerts);
d0 = sqrt(sum((info0.V(idx0, :) - model.tar.V).^2, 2));
fprintf('fit distances at P0 (mesh units): mean %.2f  median %.2f  max %.2f\n', mean(d0), median(d0), max(d0));
fprintf('check: sum(Efit.^2) = sum(d.^2)/nTar = %.6f vs %.6f\n', e0(1), sum(d0.^2) / nTar);

%% 4. After optimization
% 30 LM iterations (see t06) give the "solved" parameters to compare with.
[P1, h1] = edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 30));
[F1, ~, info1] = edsm.residual(P1, model);
[~, e1] = termEnergies(F1, info1);
fprintf('\nAfter %d LM iterations, err = %.6f\n', h1.nIter, sum(F1.^2));
fprintf('  %-8s %14s %14s\n', 'term', 'at P0', 'after LM');
for k = 1:4
    fprintf('  %-8s %14.6g %14.6g\n', names{k}, e0(k), e1(k));
end
d1 = sqrt(sum((info1.V(info1.indices, :) - model.tar.V).^2, 2));
fprintf('fit distances after LM: mean %.2f  median %.2f  max %.2f; distinct source vertices matched %d\n', ...
    mean(d1), median(d1), max(d1), numel(unique(info1.indices)));

fig = newFig(ctx, 8, 4.2);
ax = axes(fig);
b = bar(ax, 1:4, [e0; e1]', 'grouped');
b(1).FaceColor = [0.65 0.65 0.65]; b(2).FaceColor = col.lm;
set(ax, 'XTick', 1:4, 'XTickLabel', names, 'YScale', 'log');
ylim(ax, [1e-6 1e4]);
ylabel(ax, 'sum of squares  (contribution to err)');
legend(ax, {'at P0 (identity)', sprintf('after %d LM iterations', h1.nIter)}, 'Location', 'northeast');
title(ax, sprintf('err = %.1f at P0, %.1f after LM', sum(F0.^2), sum(F1.^2)));
for k = 1:4
    text(ax, k - 0.15, max(e0(k), 1e-6) * 1.9, sprintf('%.3g', e0(k)), 'HorizontalAlignment', 'center', 'FontSize', 9);
    text(ax, k + 0.15, e1(k) * 1.9, sprintf('%.3g', e1(k)), 'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', col.lm);
end
saveFig(ctx, fig, 'terms');

%% 5. How each term reacts to specific perturbations of P
% Change one thing at a time and look at which terms respond. Node iN is a
% chest node. Values are the four sums of squares at the perturbed P.
iN = 42;
Rz = @(a) [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];
scen = {};
scen(end+1, :) = {'identity (P0)', P0};
scen(end+1, :) = {'A_i = 1.2 I (scale one node)', setNodeParam(P0, n, iN, 1.2 * eye(3), [0 0 0])};
scen(end+1, :) = {'A_i = rotation by 30 deg', setNodeParam(P0, n, iN, Rz(pi / 6), [0 0 0])};
scen(end+1, :) = {'t_i = (0, 0, 50) (move one node)', setNodeParam(P0, n, iN, eye(3), [0 0 50])};
Pall = P0; Pall(6 + 9 * n + 1:end) = repmat([0; 0; 50], n, 1);
scen(end+1, :) = {'all t_i = (0, 0, 50)', Pall};
Pg = P0; Pg(4:6) = [0 0 50];
scen(end+1, :) = {'global T = (0, 0, 50)', Pg};
nS = size(scen, 1);
E = zeros(nS, 4);
for s = 1:nS
    [Fs, ~, is] = edsm.residual(scen{s, 2}, model);
    [~, E(s, :)] = termEnergies(Fs, is);
end
fprintf('\nSum of squares per term (node %d for single-node cases):\n', iN);
fprintf('  %-36s %12s %12s %12s %12s %12s\n', 'perturbation', names{:}, 'total');
for s = 1:nS
    fprintf('  %-36s %12.4g %12.4g %12.4g %12.4g %12.6f\n', scen{s, 1}, E(s, :), sum(E(s, :)));
end
fprintf('Change relative to P0:\n');
for s = 2:nS
    fprintf('  %-36s %12.4g %12.4g %12.4g %12.4g\n', scen{s, 1}, E(s, :) - E(1, :));
end

fig = newFig(ctx, 8, 4.6);
ax = axes(fig, 'Position', [0.34 0.27 0.62 0.65]);
floorV = 1e-10;
Ep = max(E, floorV);
bh = barh(ax, 1:nS, Ep, 'grouped');
tc = [col.fit; col.rot; col.smooth; col.lap];
for k = 1:4, bh(k).FaceColor = tc(k, :); bh(k).BaseValue = floorV; end
set(ax, 'YTick', 1:nS, 'YTickLabel', scen(:, 1), 'YDir', 'reverse', 'XScale', 'log');
xlim(ax, [floorV 1e4]);
xlabel(ax, 'sum of squares of the term  (no bar = 0)');
lg = legend(ax, names, 'Location', 'none', 'Orientation', 'horizontal');
lg.Position = [0.34 0.02 0.62 0.07];
title(ax, 'Which terms respond to which change');
saveFig(ctx, fig, 'perturbations');

%% 6. Per-vertex fit error on the target mesh
% Color every target vertex by its distance to the nearest deformed-source
% vertex (this distance is what Efit squares). Left: before, right: after LM.
bnd = meshBounds(model.src.V, model.tar.V);
cmax = 40;
fig = newFig(ctx, 8, 5.8);
axs = panelGrid(fig, 1, 2, 'Margins', [0.01 0.01 0.12 0.09], 'Gap', [0.02 0]);
dd = {d0, d1};
tt = {sprintf('at P0: mean %.1f, max %.1f', mean(d0), max(d0)), sprintf('after LM: mean %.1f, max %.1f', mean(d1), max(d1))};
for k = 1:2
    drawSurface(axs(k), model.tar.V, model.tar.F, 'CData', min(dd{k}, cmax));
    axs(k).Colormap = viridisMap(256); axs(k).CLim = [0 cmax];
    setBodyView(axs(k), bnd);
    panelTitle(axs(k), tt{k});
end
cb = colorbar(axs(2), 'Location', 'southoutside');
cb.Position = [0.30 0.075 0.40 0.025];
cb.Label.String = sprintf('distance to nearest deformed source vertex (mesh units; values > %d shown as %d)', cmax, cmax);
cb.FontSize = 10;
fprintf('\nfraction of target vertices farther than %d from the source: P0 %.1f%%, after LM %.1f%%\n', cmax, 100 * mean(d0 > cmax), 100 * mean(d1 > cmax));
saveFig(ctx, fig, 'fit_error');

%% 7. Figure: the correspondences, zoomed on the worst-fitting region
% Zoom on the target vertex with the largest fit distance at P0 (the lower
% left arm in the front view). Every second target vertex inside the window
% is joined (black line) to the deformed-source vertex it is matched to.
% Left: at P0. Right: after LM, where the same vertices are matched to
% much nearer source points. The matches are nearest points, not anatomical
% correspondences.
[~, vWorst] = max(d0);
cen = model.tar.V(vWorst, :) + [150 0 -60];   % window center shifted up/inward from the worst vertex
half = 260;
crop = [cen - half; cen + half];
lim = bnd + [-1; 1] * 200; lim(:, 1) = crop(:, 1); lim(:, 3) = crop(:, 3);
inWin = find(abs(model.tar.V(:, 1) - cen(1)) < half & abs(model.tar.V(:, 3) - cen(3)) < half);
pick = inWin(1:2:end);
fprintf('\nzoom window near target vertex %d (d0 = %.1f): %d target vertices inside, %d lines drawn\n', vWorst, d0(vWorst), numel(inWin), numel(pick));
infos = {info0, info1}; ttl = {'at P0', sprintf('after %d LM iterations', h1.nIter)};
fig = newFig(ctx, 8, 4.6);
axs = panelGrid(fig, 1, 2, 'Margins', [0.01 0.01 0.09 0.09], 'Gap', [0.02 0]);
for k = 1:2
    Vs = infos{k}.V; ix = infos{k}.indices;
    drawSurface(axs(k), Vs, model.src.F, 'Color', col.source, 'Alpha', 0.3);
    drawSurface(axs(k), model.tar.V, model.tar.F, 'Color', col.target, 'Alpha', 0.3);
    A = Vs(ix(pick), :) + [0 -30 0]; B = model.tar.V(pick, :) + [0 -30 0];
    X = nan(3 * numel(pick), 3); X(1:3:end, :) = A; X(2:3:end, :) = B;
    plot3(axs(k), X(:, 1), X(:, 2), X(:, 3), '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 1);
    scatter3(axs(k), A(:, 1), A(:, 2), A(:, 3), 10, col.source, 'filled');
    scatter3(axs(k), B(:, 1), B(:, 2), B(:, 3), 10, col.target, 'filled');
    setBodyView(axs(k), crop, 'Limits', lim);
    panelTitle(axs(k), ttl{k});
end
annotation(fig, 'textbox', [0 0 1 0.07], 'String', 'blue = deformed source, orange = target; lines: target vertex -> matched source vertex', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11);
saveFig(ctx, fig, 'correspondences');

tutorialDone(ctx);
