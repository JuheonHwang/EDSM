%% t01_ed_model  The Embedded Deformation graph and its parameters
% Shows what the 606 numbers that the optimizers search over mean: a sparse
% set of ED nodes on the source mesh, each carrying an affine matrix A_i and
% a translation t_i, blended onto every vertex by per-vertex weights.
%
%   v'_j = R * ( sum_k w_jk * ( A_k (v_j - g_k) + g_k + t_k ) - G0 ) + G0 + T
%
% (k runs over the 4 nodes that influence vertex j; g_k are the node positions,
% G0 is the centroid of the source; R = rodrigues(r), T global translation.)
clearvars; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'helpers'));
ctx = tutorialInit('t01_ed_model');
col = tutorialColors();

%% 1. Model and ED nodes
% About 1% of the source vertices (random, seed 1) become ED nodes. The node
% positions g_i are simply those vertices.
[model, P0] = loadTutorialModel(ctx);
n = model.nNodes;
fprintf('nNodes = %d (= round(%d * sampleRate %.2f))\n', n, model.nVerts, model.cfg.sampleRate);
fprintf('first 5 nodes: source vertex index and position g_i\n');
for i = 1:5
    fprintf('  node %2d  vertex %4d  g = [%8.1f %8.1f %8.1f]\n', i, model.nodeIdx(i), model.g0(:, i));
end
fprintf('g0 equals src.V(nodeIdx,:)'': %d\n', isequal(model.g0, model.src.V(model.nodeIdx, :)'));

%% 2. Node graph: neighbors of each node
% idxNodeNeighbor(:,i) lists the 6 nearest nodes of node i (geodesic distance
% along the surface), the first being node i itself. The smoothness energy
% couples each node to its 5 other neighbors.
nb = model.idxNodeNeighbor;
fprintf('idxNodeNeighbor is %d x %d; first column is the node itself: %d\n', size(nb, 1), size(nb, 2), all(nb(1, :) == 1:n));
E = [repelem((1:n)', size(nb, 1) - 1, 1), reshape(nb(2:end, :), [], 1)];   % directed edges i -> j
Eu = unique(sort(E, 2), 'rows');
fprintf('directed neighbor links: %d, undirected edges: %d\n', size(E, 1), size(Eu, 1));
g = model.g0';
% Count mesh vertices that have no valid ED node (NaN neighbors).
fprintf('nodes with x < -650 (the floor slab at the feet, approximate threshold): %d of %d\n', sum(g(:, 1) < -650), n);
nInvalid = sum(any(isnan(model.idxNeighbor), 1));
fprintf('vertices without valid ED neighbors (kept at the source position): %d of %d\n', nInvalid, model.nVerts);

bnd = meshBounds(model.src.V, model.tar.V);
fig = newFig(ctx, 8, 5.4);
axs = panelGrid(fig, 1, 2, 'Margins', [0.01 0.01 0.08 0.09], 'Gap', [0.02 0]);
views = {'front', 'side'};
shift = {[0 -25 0], [0 0 25]};
for k = 1:2
    drawSurface(axs(k), model.src.V, model.src.F, 'Color', [0.86 0.86 0.86], 'Alpha', 0.30);
    X = nan(3 * size(Eu, 1), 3);
    X(1:3:end, :) = g(Eu(:, 1), :) + shift{k};
    X(2:3:end, :) = g(Eu(:, 2), :) + shift{k};
    plot3(axs(k), X(:, 1), X(:, 2), X(:, 3), '-', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.8);
    scatter3(axs(k), g(:, 1) + shift{k}(1), g(:, 2) + shift{k}(2), g(:, 3) + shift{k}(3), 28, col.gd, 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    setBodyView(axs(k), bnd, 'View', views{k});
    panelTitle(axs(k), sprintf('%s view', views{k}));
end
annotation(fig, 'textbox', [0 0.0 1 0.07], 'String', sprintf('%d ED nodes (orange dots) and %d neighbor edges (dark lines) on the source mesh', n, size(Eu, 1)), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11);
saveFig(ctx, fig, 'graph');

%% 3. Per-vertex blending weights
% Each vertex is influenced by its 4 nearest nodes with weight 1 - d/d5
% (d = geodesic distance to the node, d5 = distance to the 5th nearest node),
% normalized to sum 1. Build the sparse matrix W (nVerts x nNodes) from
% idxNeighbor / distNeighbor and look at one column: the weight of one node.
K = model.cfg.nVertexNeighbor;
valid = ~any(isnan(model.idxNeighbor), 1);
iv = find(valid); iv = iv(:);
rowsI = repmat(iv, K, 1);
colsJ = reshape(model.idxNeighbor(:, valid)', [], 1);
vals = reshape(model.distNeighbor(:, valid)', [], 1);
W = sparse(rowsI, colsJ, vals, model.nVerts, n);
rs = full(sum(W, 2));
fprintf('row sums of W over valid vertices: min %.6f max %.6f\n', min(rs(valid)), max(rs(valid)));
nzRow = full(sum(W(valid, :) > 0, 2));
fprintf('valid vertices with 4 / 3 / 2 / 1 nonzero weights: %d / %d / %d / %d (nVertexNeighbor = %d)\n', sum(nzRow == 4), sum(nzRow == 3), sum(nzRow == 2), sum(nzRow == 1), K);
nSupp = full(sum(W > 0, 1));
fprintf('vertices influenced per node: min %d  median %d  max %d\n', min(nSupp), median(nSupp), max(nSupp));
% Zero weights can occur: the 4th nearest node has weight 1 - d4/d5, which is 0 for ties.

% Pick three nodes that are visible from the front: head, chest and a node on the thigh.
[~, vHead] = max(model.src.V(:, 1));
[~, iHead] = min(sum((g - model.src.V(vHead, :)).^2, 2));
chest = mean(model.src.V(model.src.V(:, 1) > 450 & model.src.V(:, 1) < 700, :), 1);
[~, iChest] = min(sum((g - chest).^2, 2));
thigh = mean(model.src.V(model.src.V(:, 1) > -100 & model.src.V(:, 1) < 100, :), 1);
[~, iThigh] = min(sum((g - thigh).^2, 2));
sel = [iHead iChest iThigh];
names = {'head', 'chest', 'thigh'};
for k = 1:3
    fprintf('selected node %2d (%s): position [%7.1f %7.1f %7.1f], influences %d vertices\n', sel(k), names{k}, g(sel(k), :), nSupp(sel(k)));
end

fig = newFig(ctx, 8, 5.6);
axs = panelGrid(fig, 1, 3, 'Margins', [0.01 0.01 0.12 0.09], 'Gap', [0.01 0]);
cm = [0.88 0.88 0.88; viridisMap(255)];
for k = 1:3
    wk = full(W(:, sel(k)));
    drawSurface(axs(k), model.src.V, model.src.F, 'CData', wk);
    axs(k).Colormap = cm; axs(k).CLim = [0 1];
    setBodyView(axs(k), bnd);
    scatter3(axs(k), g(sel(k), 1), g(sel(k), 2) - 25, g(sel(k), 3), 70, 'w', 'filled', 'MarkerEdgeColor', 'r', 'LineWidth', 2);
    panelTitle(axs(k), sprintf('node %d (%s)', sel(k), names{k}));
end
cb = colorbar(axs(3), 'Location', 'southoutside');
cb.Position = [0.30 0.075 0.40 0.025];
cb.Label.String = 'blending weight of the node (gray = 0)';
cb.FontSize = 10;
saveFig(ctx, fig, 'weights');

%% 4. Parameter vector layout
% P = [ r(3); T(3); A_1(:) ... A_n(:) (9n); t_1 ... t_n (3n) ], 6 + 12n entries.
% A_i is stored column-major (A(:)). The identity deformation P0 therefore
% has zeros except for ones at the diagonal slots of each A_i.
fprintf('numel(P0) = %d = 6 + 12*%d\n', numel(P0), n);
fprintf('r = P(1:3), T = P(4:6), A = P(7:%d), t = P(%d:%d)\n', 6 + 9 * n, 7 + 9 * n, numel(P0));
iA = @(i) 6 + 9 * (i - 1) + (1:9);
a3 = iA(3);
fprintf('A_3 occupies P(%d:%d); t_3 occupies P(%d:%d)\n', a3(1), a3(end), 6 + 9 * n + 3 * 2 + 1, 6 + 9 * n + 3 * 3);
nz = find(P0(1:33));
fprintf('nonzero entries of P0(1:33): %s\n', mat2str(nz'));

% Round trip with random values: encode -> decode -> compare.
rng(3);
Rrand = edsm.rodrigues(0.7 * randn(1, 3));
Trand = randn(1, 3);
Arand = randn(3, 3, n);
trand = randn(3, n);
Pr = edsm.encodeParam(Rrand, Trand, Arand, trand);
[R2, T2, A2, t2] = edsm.decodeParam(Pr, n);
fprintf('round trip max error: R %.1e  T %.1e  A %.1e  t %.1e\n', ...
    max(abs(R2(:) - Rrand(:))), max(abs(T2 - Trand)), max(abs(A2(:) - Arand(:))), max(abs(t2(:) - trand(:))));
a2 = iA(2);
fprintf('A_2 = P(%d:%d) reshaped to 3x3 equals A(:,:,2): %d\n', a2(1), a2(end), isequal(reshape(Pr(a2), 3, 3), Arand(:, :, 2)));

fig = newFig(ctx, 8, 5);
ax1 = axes(fig, 'Position', [0.06 0.58 0.90 0.34]); hold(ax1, 'on'); axis(ax1, [0 1 0 1]); axis(ax1, 'off');
segs = {'r', 'rotation vector of R', '3', '1:3', col.lm; ...
        'T', 'global translation', '3', '4:6', col.lmEye; ...
        'A_1 ... A_n', 'affine matrices, column-major', sprintf('9n = %d', 9 * n), sprintf('7:%d', 6 + 9 * n), col.rot; ...
        't_1 ... t_n', 'node translations', sprintf('3n = %d', 3 * n), sprintf('%d:%d', 7 + 9 * n, numel(P0)), col.smooth};
w = [0.10 0.10 0.50 0.30]; x = 0;
for k = 1:4
    rectangle(ax1, 'Position', [x 0.35 w(k) 0.40], 'FaceColor', min(1, segs{k, 5} * 0.45 + 0.55), 'EdgeColor', col.text);
    text(ax1, x + w(k) / 2, 0.55, segs{k, 1}, 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    text(ax1, x + w(k) / 2, 0.20, sprintf('P(%s)', segs{k, 4}), 'HorizontalAlignment', 'center', 'FontSize', 10);
    text(ax1, x + w(k) / 2, 0.05, sprintf('%s entries', segs{k, 3}), 'HorizontalAlignment', 'center', 'FontSize', 10);
    x = x + w(k);
end
text(ax1, 0.5, 0.95, sprintf('Parameter vector P (%d entries, n = %d nodes)   [widths not to scale]', numel(P0), n), ...
    'HorizontalAlignment', 'center', 'FontSize', 12);
ax2 = axes(fig, 'Position', [0.09 0.12 0.88 0.36]);
idx = 1:33;
grp = [ones(1, 3), 2 * ones(1, 3), 3 * ones(1, 27)];
cl = [col.lm; col.lmEye; col.rot];
hold(ax2, 'on');
for gI = 1:3
    m = grp == gI;
    stem(ax2, idx(m), P0(idx(m)), 'filled', 'Color', cl(gI, :), 'MarkerFaceColor', cl(gI, :));
end
xlabel(ax2, 'index in P'); ylabel(ax2, 'P0 value');
title(ax2, 'First 33 entries of the identity P0: r, T, then A_1, A_2, A_3 (each 9 values, column-major)');
ax2.Title.FontSize = 10.5;
ylim(ax2, [-0.2 1.3]); xlim(ax2, [0 34]); xticks(ax2, [1 4 7 16 25 34 - 1]);
legend(ax2, {'r', 'T', 'A_i'}, 'Location', 'eastoutside', 'FontSize', 9);
saveFig(ctx, fig, 'layout');

%% 5. Hand-made deformations through edsm.deform
% Start from P0 and change a single node: its translation t_i (the patch
% moves), its matrix A_i (the patch rotates/scales about g_i). deform
% blends this node's effect into the vertices with weights W(:,i).
V0 = edsm.deform(P0, model);
fprintf('deform(P0) vs source vertices: max |diff| = %.2e (identity)\n', max(abs(V0(:) - model.src.V(:))));
iN = iChest;
setLocal = @(Aval, tval) setNodeParam(P0, n, iN, Aval, tval);
Pt = setLocal(eye(3), [0; 0; 150]);                          % translate node by 150 along z
Pr = setLocal(edsm.rodrigues([0 pi / 4 0]), [0; 0; 0]);      % rotate A_i by 45 deg about y (the view axis)
Ps = setLocal(1.6 * eye(3), [0; 0; 0]);                      % scale A_i by 1.6
cases = {'original', P0; 't_i = (0, 0, 150)', Pt; 'A_i = 45 deg rotation about y', Pr; 'A_i = 1.6 I', Ps};
Vc = cell(1, 4); disp_ = cell(1, 4);
for k = 1:4
    Vc{k} = edsm.deform(cases{k, 2}, model);
    disp_{k} = sqrt(sum((Vc{k} - model.src.V).^2, 2));
    fprintf('%-32s: %4d vertices move (>1e-6), max displacement %.1f, influence region of node %d has %d vertices\n', ...
        cases{k, 1}, sum(disp_{k} > 1e-6), max(disp_{k}), iN, nSupp(iN));
end
dmax = max(cellfun(@max, disp_));
half = 380;
crop = [g(iN, :) - half; g(iN, :) + half];
fig = newFig(ctx, 8, 8.2);
axs = panelGrid(fig, 2, 2, 'Margins', [0.01 0.01 0.07 0.06], 'Gap', [0.01 0.05]);
cm = viridisMap(256);
for k = 1:4
    ax = axs(ceil(k / 2), mod(k - 1, 2) + 1);
    if k == 1
        drawSurface(ax, Vc{k}, model.src.F, 'Color', col.neutral);
    else
        drawSurface(ax, Vc{k}, model.src.F, 'CData', disp_{k});
        ax.Colormap = cm; ax.CLim = [0 dmax];
    end
    lim = bnd + [-1; 1] * 200;                 % full body in y, the crop window in x and z:
lim(:, 1) = crop(:, 1); lim(:, 3) = crop(:, 3);   % patches are clipped to the axes limits
    setBodyView(ax, crop, 'Limits', lim);
    scatter3(ax, g(iN, 1), g(iN, 2) - 40, g(iN, 3), 60, 'w', 'filled', 'MarkerEdgeColor', 'r', 'LineWidth', 2);
    panelTitle(ax, cases{k, 1});
end
cb = colorbar(axs(2, 2), 'Location', 'southoutside');
cb.Position = [0.30 0.025 0.40 0.015];
cb.Label.String = sprintf('vertex displacement (mesh units), node %d circled', iN);
cb.FontSize = 10;
saveFig(ctx, fig, 'local_deform');

% Global rotation / translation: r and T act on the whole body after blending.
Rg = edsm.rodrigues([0 0.35 0]);
Pg = edsm.encodeParam(Rg, [0 0 100], repmat(eye(3), [1 1 n]), zeros(3, n));
Vg = edsm.deform(Pg, model);
fprintf('global: r = %s, T = %s -> max displacement %.1f, centroid moves by %s\n', mat2str([0 0.35 0]), mat2str([0 0 100]), ...
    max(sqrt(sum((Vg - model.src.V).^2, 2))), mat2str(mean(Vg - model.src.V, 1), 4));
fig = newFig(ctx, 8, 5.4);
axs = panelGrid(fig, 1, 2, 'Margins', [0.01 0.01 0.02 0.09], 'Gap', [0.02 0]);
bnd2 = meshBounds(model.src.V, Vg);
drawSurface(axs(1), model.src.V, model.src.F, 'Color', col.neutral); setBodyView(axs(1), bnd2); panelTitle(axs(1), 'original');
drawSurface(axs(2), model.src.V, model.src.F, 'Color', [0.9 0.9 0.9], 'Alpha', 0.35);
drawSurface(axs(2), Vg, model.src.F, 'Color', col.source); setBodyView(axs(2), bnd2);
panelTitle(axs(2), 'r = (0, 0.35, 0) rad, T = (0, 0, 100)');
saveFig(ctx, fig, 'global_deform');

tutorialDone(ctx);
