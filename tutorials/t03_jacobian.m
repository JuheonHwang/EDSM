%% t03_jacobian  Structure and correctness of the analytic Jacobian
% J = dF/dP has one row per residual and one column per parameter. Every
% second-order method here (Gauss-Newton, Levenberg-Marquardt, CG on J'J)
% depends on it, so it is worth knowing (a) where J is nonzero, (b) how to
% verify it against finite differences, and (c) its rank deficiency.
% The finite-difference check is run for the corrected Jacobian and for the
% legacy one (cfg.legacyJacobian = true) that the original scripts used.
clearvars; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'helpers'));
ctx = tutorialInit('t03_jacobian');
col = tutorialColors();

%% 1. Jacobian at the starting point and its sparsity pattern
% Rows are grouped by energy term (info.rows), columns by parameter group:
% r (1:3), T (4:6), all A_i (7:456) and all t_i (457:606).
[model, P0] = loadTutorialModel(ctx);
n = model.nNodes;
[F0, J0, info0] = edsm.residual(P0, model);
fprintf('J at P0: %d x %d, nnz = %d (%.3f%% dense)\n', size(J0, 1), size(J0, 2), nnz(J0), 100 * nnz(J0) / numel(J0));
cg = struct('r', 1:3, 'T', 4:6, 'A', 7:(6 + 9 * n), 't', (7 + 9 * n):numel(P0));
rg = info0.rows;
rowNames = {'fit', 'rot1', 'rot2', 'smooth', 'lap'};
colNames = fieldnames(cg)';
fprintf('\nnonzeros per block (rows: terms, columns: parameter groups)\n');
fprintf('  %-7s', ''); fprintf('%10s', colNames{:}); fprintf('\n');
for i = 1:5
    fprintf('  %-7s', rowNames{i});
    for j = 1:4
        blk = J0(rg.(rowNames{i})(1):rg.(rowNames{i})(2), cg.(colNames{j}));
        fprintf('%10d', nnz(blk));
    end
    fprintf('\n');
end

fig = newFig(ctx, 8, 6);
zoomRows = rg.rot1(1):rg.smooth(2);
panels = {1:size(J0, 1), zoomRows};
ttl = {sprintf('all %d rows', size(J0, 1)), sprintf('zoom: rows %d-%d (rot1, rot2, smooth)', zoomRows(1), zoomRows(end))};
pos = {[0.17 0.12 0.32 0.78], [0.66 0.12 0.31 0.78]};
for k = 1:2
    ax = axes(fig, 'Position', pos{k});
    rr = panels{k};
    axes(ax); spy(J0(rr, :), 'k', 1); %#ok<LAXES>
    ax.DataAspectRatioMode = 'auto'; ax.PlotBoxAspectRatioMode = 'auto';
    hold(ax, 'on');
    xb = [3.5 6.5 6.5 + 9 * n];
    for x = xb, xline(ax, x, '--', 'Color', col.rot, 'LineWidth', 1); end
    bounds = [rg.fit(2) rg.rot1(2) rg.rot2(2) rg.smooth(2)] - rr(1) + 1.5;
    for yb = bounds(bounds > 0 & bounds < numel(rr)), yline(ax, yb, '--', 'Color', col.lm, 'LineWidth', 1); end
    xlabel(ax, 'parameter group (columns)');
    title(ax, ttl{k}); ax.Title.FontSize = 10.5;
    ax.XTick = [4.5 231 531]; ax.XTickLabel = {'r | T', 'A', 't'};
    ax.YGrid = 'off'; ax.XGrid = 'off';
    if k == 1
        ax.YTick = [mean([1 rg.fit(2)]), mean([rg.rot1(1) rg.smooth(2)]), mean(rg.lap)];
        ax.YTickLabel = {'fit', 'rot+smooth', 'lap'};
    else
        ax.YTick = [mean(rg.rot1), mean(rg.rot2), mean(rg.smooth)] - rr(1) + 1;
        ax.YTickLabel = {'rot1', 'rot2', 'smooth'};
    end
    ylabel(ax, 'residual term (rows)');
end
annotation(fig, 'textbox', [0 0.94 1 0.05], 'String', 'Sparsity of J at P0  (dashed: term boundaries blue, parameter-group boundaries orange)', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11);
saveFig(ctx, fig, 'sparsity');

%% 2. Finite-difference check of every block
% A random point near P0 (so that A_i, t_i, r, T are generic) is used, since
% at P0 many derivative terms vanish. numericJacobian takes central
% differences holding the nearest-neighbor correspondences fixed, exactly
% like the analytic Jacobian. All 606 columns are checked (1212 residual
% evaluations, a few seconds).
rng(2);
P1 = P0 + [0.02 * randn(3, 1); 5 * randn(3, 1); 0.03 * randn(9 * n, 1); 3 * randn(3 * n, 1)];
cols = 1:numel(P0);
tic;
Jn = edsm.numericJacobian(P1, model, struct('cols', cols));
fprintf('\nfinite differences: %d columns, %d residual evaluations, %.1f s\n', numel(cols), 2 * numel(cols), toc);
modes = {'corrected', 'legacy'};
Err = nan(5, 4, 2); Rel = nan(5, 4, 2);
for m = 1:2
    modelM = edsm.buildModel(model.src, model.tar, setfield(model.cfg, 'legacyJacobian', m == 2)); %#ok<SFLD>
    [~, Ja] = edsm.residual(P1, modelM);
    Ja = Ja(:, cols);
    for i = 1:5
        ri = rg.(rowNames{i})(1):rg.(rowNames{i})(2);
        for j = 1:4
            cj = ismember(cols, cg.(colNames{j}));
            if ~any(cj), continue, end
            D = Ja(ri, cj) - Jn(ri, cj);
            Err(i, j, m) = max(abs(D(:)));
            sc = max(max(abs(Jn(ri, cj)), [], 'all'), max(abs(Ja(ri, cj)), [], 'all'));
            if sc > 0, Rel(i, j, m) = Err(i, j, m) / max(sc, 1); end
        end
    end
    fprintf('\n%s Jacobian: max |J_analytic - J_numeric| per block\n', modes{m});
    fprintf('  %-7s', ''); fprintf('%12s', colNames{:}); fprintf('\n');
    for i = 1:5
        fprintf('  %-7s', rowNames{i}); fprintf('%12.2e', Err(i, :, m)); fprintf('\n');
    end
    fprintf('%s Jacobian: same, relative to max(1, largest entry of the block) (NaN = block is zero in both)\n', modes{m});
    fprintf('  %-7s', ''); fprintf('%12s', colNames{:}); fprintf('\n');
    for i = 1:5
        fprintf('  %-7s', rowNames{i}); fprintf('%12.2e', Rel(i, :, m)); fprintf('\n');
    end
end

fig = newFig(ctx, 8, 3.8);
cm = viridisMap(256);
for m = 1:2
    ax = axes(fig, 'Position', [0.13 + 0.40 * (m - 1), 0.17, 0.33, 0.68]);
    L = log10(max(Rel(:, :, m), 1e-16));
    L(isnan(Rel(:, :, m))) = NaN;
    im = imagesc(ax, L, 'AlphaData', ~isnan(L));
    ax.Colormap = flipud(cm); ax.CLim = [-12 0]; ax.Color = [0.93 0.93 0.93];
    set(ax, 'XTick', 1:4, 'XTickLabel', colNames, 'YTick', 1:5, 'YTickLabel', rowNames, 'YDir', 'reverse', 'XGrid', 'off', 'YGrid', 'off');
    for i = 1:5
        for j = 1:4
            if isnan(L(i, j))
                txt = '0'; tc = [0.4 0.4 0.4];
            else
                txt = sprintf('%.0e', Rel(i, j, m)); tc = 'k'; if L(i, j) > -3.5, tc = 'w'; end
            end
            text(ax, j, i, txt, 'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', tc);
        end
    end
    title(ax, [modes{m} ' Jacobian']);
    xlabel(ax, 'parameter group'); if m == 1, ylabel(ax, 'residual term'); end
end
cb = colorbar(ax, 'Position', [0.90 0.17 0.015 0.68]);
cb.Label.String = 'log_{10} relative error'; cb.FontSize = 9;
saveFig(ctx, fig, 'fd_check');

%% 3. The rank deficiency of J at P0
% Singular values of J (via a sparse QR to avoid forming the 30830 x 606
% dense matrix). Six are numerically zero: moving T and all t_i oppositely
% does nothing (3 directions), and a global rotation of the body can be
% undone by counter-rotating every A_i and adjusting t_i (3 more). J'J is
% therefore singular and the plain normal equations cannot be solved.
Rq = qr(J0, 0);
[~, S, Vr] = svd(full(Rq));
sv = diag(S);
fprintf('\nsingular values of J at P0: largest %.4g, 7th smallest %.4g, 6 smallest: %s\n', sv(1), sv(end - 6), mat2str(sv(end - 5:end)', 3));
fprintf('gap: sv(end-6) / sv(end-5) = %.3g;  condition number of the nonzero part sv(1)/sv(end-6) = %.3g\n', sv(end - 6) / sv(end - 5), sv(1) / sv(end - 6));
fprintf('numerical rank (tol = eps*max(size(J))*sv(1) = %.2e): %d of %d\n', eps * max(size(J0)) * sv(1), sum(sv > eps * max(size(J0)) * sv(1)), numel(sv));

% The six null directions built by hand.
G0 = model.G0(:); g = model.g0;
Vn = zeros(numel(P0), 6);
skew = @(a) [0 -a(3) a(2); a(3) 0 -a(1); -a(2) a(1) 0];
for k = 1:3
    e = zeros(3, 1); e(k) = 1;
    v = zeros(numel(P0), 1);
    v(3 + k) = 1;                                                % dT = e_k
    v(cg.t) = reshape(repmat(-e, 1, n), [], 1);                  % dt_i = -e_k
    Vn(:, k) = v;
    v = zeros(numel(P0), 1);
    v(k) = 1;                                                    % dr = e_k
    v(cg.A) = reshape(repmat(-skew(e), [1 1 n]), [], 1);         % dA_i = -[e_k]x
    v(cg.t) = reshape(-skew(e) * (g - G0), [], 1);               % dt_i = -[e_k]x (g_i - G0)
    Vn(:, 3 + k) = v;
end
fprintf('\nhand-built null directions: |J v| / |v| (compare with sv(1) = %.3g)\n', sv(1));
lbl = {'dT = e_x, dt_i = -e_x', 'dT = e_y, dt_i = -e_y', 'dT = e_z, dt_i = -e_z', ...
       'dr = e_x, dA_i = -[e_x]x, dt_i = -[e_x]x(g_i-G0)', 'dr = e_y, ... (same with e_y)', 'dr = e_z, ... (same with e_z)'};
for k = 1:6
    fprintf('  %-52s %.3e\n', lbl{k}, norm(J0 * Vn(:, k)) / norm(Vn(:, k)));
end
ang = subspace(Vn, Vr(:, end - 5:end));
fprintf('angle between span(hand-built) and the 6 smallest right singular vectors: %.2e rad\n', ang);

fig = newFig(ctx, 8, 3.8);
ax = axes(fig, 'Position', [0.08 0.17 0.40 0.74]);
semilogy(ax, sv, '-', 'Color', col.lm); hold(ax, 'on');
semilogy(ax, numel(sv) - 5:numel(sv), max(sv(end - 5:end), 1e-18), 'o', 'Color', col.gd, 'MarkerFaceColor', col.gd);
xlabel(ax, 'index (sorted)'); ylabel(ax, 'singular value of J'); title(ax, 'All 606 singular values');
ylim(ax, [1e-18 1e4]);
ax = axes(fig, 'Position', [0.58 0.17 0.39 0.74]);
ii = numel(sv) - 19:numel(sv);
semilogy(ax, ii, max(sv(ii), 1e-18), '-o', 'Color', col.lm, 'MarkerFaceColor', col.lm, 'MarkerSize', 4); hold(ax, 'on');
semilogy(ax, ii(end - 5:end), max(sv(ii(end - 5:end)), 1e-18), 'o', 'Color', col.gd, 'MarkerFaceColor', col.gd);
xlabel(ax, 'index (sorted)'); ylabel(ax, 'singular value of J'); title(ax, 'Last 20: six values are zero to rounding');
ylim(ax, [1e-18 1e4]);
legend(ax, {'singular values', 'null space (6)'}, 'Location', 'west', 'FontSize', 9);
saveFig(ctx, fig, 'singular_values');

tutorialDone(ctx);
