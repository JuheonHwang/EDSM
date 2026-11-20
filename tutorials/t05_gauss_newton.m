%% t05_gauss_newton  Gauss-Newton: a great first step, then divergence
% Gauss-Newton replaces F(P+d) by its linearization F + J d and minimizes
% ||F + J d||^2 over d, i.e. solves the normal equations J'J d = -J'F. It has
% no step control: edsm.solvers.gaussNewton takes the minimum-norm
% least-squares step with lsqminnorm and applies it as is.
clearvars; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'helpers'));
ctx = tutorialInit('t05_gauss_newton');
col = tutorialColors();

%% 1. Normal equations vs lsqminnorm at P0
% J'J has a 6-dimensional null space at P0 (t03), so J'J \ (-J'F) has no
% unique solution. lsqminnorm picks the one of minimum norm.
[model, P0] = loadTutorialModel(ctx);
n = model.nNodes;
[F0, J0] = edsm.residual(P0, model);
JtJ = J0' * J0; g0 = J0' * F0;
err0 = sum(F0.^2);
lastwarn('');
dBackslash = JtJ \ (-g0);
[wmsg, wid] = lastwarn();
if isempty(wmsg), wmsg = '(none issued)'; end
fprintf('JtJ \\ (-g): MATLAB warning: %s %s\n', wmsg, wid);
fprintf('rcond(JtJ) = %.3e\n', rcond(full(JtJ)));
dMinNormJ = lsqminnorm(J0, -F0);                 % what the solver does
dMinNormN = lsqminnorm(JtJ, -g0);                % same, through the normal equations
dPinv = pinv(full(JtJ), 1e-9 * max(eig(full(JtJ)))) * (-g0);   % truncated pseudo-inverse
steps = {'JtJ \ (-g)', dBackslash; 'lsqminnorm(J, -F)', dMinNormJ; 'lsqminnorm(JtJ, -g)', dMinNormN; 'pinv(JtJ, tol) * (-g)', dPinv};
fprintf('\nstep d at P0            |d|           ||F+Jd||^2 (model)   err(P0 + d) (actual)\n');
for k = 1:size(steps, 1)
    d = steps{k, 2};
    Fn = edsm.residual(P0 + d, model);
    fprintf('  %-22s %-12.4g  %-18.6g  %.6g\n', steps{k, 1}, norm(d), sum((F0 + J0 * d).^2), sum(Fn.^2));
end
fprintf('\n|dMinNormJ - dMinNormN| / |dMinNormJ| = %.2e\n', norm(dMinNormJ - dMinNormN) / norm(dMinNormJ));
fprintf('|dMinNormJ - dPinv| / |dMinNormJ|     = %.2e\n', norm(dMinNormJ - dPinv) / norm(dMinNormJ));
fprintf('err(P0) = %.6f;  linear model predicts err = %.6f after the minimum-norm step\n', err0, sum((F0 + J0 * dMinNormJ).^2));

%% 2. Run Gauss-Newton and record every iterate
% The callback receives P after each iteration; we store them to re-evaluate
% the linear model afterwards. maxIter 12 is enough to see what happens.
store = containers.Map('KeyType', 'double', 'ValueType', 'any');   % handle object, filled by the callback
cb = @(it, P, h) putIterate(store, it, P);
[~, hist] = edsm.solvers.gaussNewton(P0, model, struct('maxIter', 12, 'callback', cb));
nIt = hist.nIter;
gnTrace = P0(:);
for k = 1:nIt, gnTrace(:, k + 1) = store(k); end
fprintf('\nGauss-Newton: %d iterations, stop reason "%s"\n', nIt, hist.stopReason);
fprintf('iter  err               |step|         \n');
fprintf('%4d  %-16.6g\n', 0, hist.err(1));
for k = 1:nIt
    fprintf('%4d  %-16.6g  %.4g\n', k, hist.err(k + 1), hist.stepNorm(k));
end

%% 3. Predicted vs actual error
% At every iterate P_k the linear model predicts err_pred = ||F_k + J_k d||^2
% for the step d = P_{k+1} - P_k. rho = (err_k - err_actual) / (err_k - err_pred)
% is the fraction of the predicted decrease that really happened (1 = model
% is exact, <= 0 means the error did not decrease).
pred = nan(nIt, 1); act = hist.err(2:end); 
for k = 1:nIt
    [Fk, Jk] = edsm.residual(gnTrace(:, k), model);
    d = gnTrace(:, k + 1) - gnTrace(:, k);
    pred(k) = sum((Fk + Jk * d).^2);
end
rho = (hist.err(1:nIt) - act) ./ (hist.err(1:nIt) - pred);
fprintf('\niter   err_before        predicted        actual           rho\n');
for k = 1:nIt
    fprintf('%4d   %-16.6g  %-15.6g  %-15.6g  %.4g\n', k, hist.err(k), pred(k), act(k), rho(k));
end

fig = newFig(ctx, 8, 3.8);
ax = axes(fig, 'Position', [0.09 0.17 0.40 0.74]);
semilogy(ax, 0:nIt, hist.err, '-o', 'Color', col.gn, 'MarkerFaceColor', col.gn, 'MarkerSize', 4);
xlabel(ax, 'iteration'); ylabel(ax, 'error  \Sigma F_k^2'); title(ax, 'Gauss-Newton error');
ax = axes(fig, 'Position', [0.58 0.17 0.39 0.74]);
semilogy(ax, 1:nIt, hist.stepNorm, '-o', 'Color', col.gn, 'MarkerFaceColor', col.gn, 'MarkerSize', 4);
xlabel(ax, 'iteration'); ylabel(ax, '||step||_2'); title(ax, 'Step length');
saveFig(ctx, fig, 'convergence');

fig = newFig(ctx, 8, 4);
ax = axes(fig, 'Position', [0.10 0.16 0.87 0.76]);
semilogy(ax, 1:nIt, hist.err(1:nIt), '--s', 'Color', [0.4 0.4 0.4], 'MarkerSize', 5, 'DisplayName', 'error before the step'); hold(ax, 'on');
semilogy(ax, 1:nIt, pred, '-o', 'Color', col.lm, 'MarkerFaceColor', col.lm, 'MarkerSize', 5, 'DisplayName', 'predicted by the linear model');
semilogy(ax, 1:nIt, act, '-o', 'Color', col.gn, 'MarkerFaceColor', col.gn, 'MarkerSize', 5, 'DisplayName', 'actual after the step');
xlabel(ax, 'Gauss-Newton iteration'); ylabel(ax, 'error  \Sigma F_k^2');
title(ax, 'The linear model is optimistic from the start');
legend(ax, 'Location', 'northwest'); xlim(ax, [1 nIt]);
saveFig(ctx, fig, 'predicted_vs_actual');

%% 4. Why the steps blow up: small singular values
% With J = U S V', the minimum-norm step is d = - sum_i (u_i'F / s_i) v_i.
% Directions with small singular value s_i (weakly constrained parameter
% combinations) get a huge coefficient unless u_i'F is tiny as well. We
% compute |u_i'F|, s_i and |coefficient| at P0 and at a later iterate.
whichIt = [0 min(3, nIt - 1)];
fig = newFig(ctx, 8, 3.9);
for q = 1:2
    Pk = gnTrace(:, whichIt(q) + 1);
    [Fk, Jk] = edsm.residual(Pk, model);
    [Cq, Rq] = qr(Jk, Fk, 0);
    [Ur, S, ~] = svd(full(Rq));
    s = diag(S); uf = Ur' * Cq;
    keep = s > eps * max(size(Jk)) * s(1);
    coef = abs(uf(keep)) ./ s(keep);
    dnorm = sqrt(sum(coef.^2));
    dsolver = lsqminnorm(Jk, -Fk);
    fprintf('\niterate %d: err %.6g, %d of %d singular values above rank tolerance, smallest kept %.3e, largest %.3e\n', ...
        whichIt(q), sum(Fk.^2), sum(keep), numel(s), min(s(keep)), s(1));
    fprintf('  |step| from SVD formula = %.4g, from lsqminnorm = %.4g\n', dnorm, norm(dsolver));
    [mx, im] = max(coef);
    sk = s(keep);
    fprintf('  largest coefficient %.4g belongs to s = %.3e (|u''F| = %.3e); share of |step|^2 from the 10 smallest s: %.1f%%\n', ...
        mx, sk(im), abs(uf(im)), 100 * sum(sort(coef(end - 9:end)).^2) / dnorm^2);
    ax = axes(fig, 'Position', [0.09 + 0.47 * (q - 1), 0.17, 0.38, 0.72]);
    loglog(ax, sk, abs(uf(keep)), '.', 'Color', col.lm, 'MarkerSize', 10, 'DisplayName', '|u_i^T F|'); hold(ax, 'on');
    loglog(ax, sk, coef, '.', 'Color', col.gn, 'MarkerSize', 10, 'DisplayName', '|u_i^T F| / s_i  (step coefficient)');
    xlim(ax, [min(sk) / 3, max(sk) * 3]); xticks(ax, 10.^(-5:1:3));
    xlabel(ax, 'singular value s_i of J'); ylabel(ax, 'magnitude');
    title(ax, sprintf('iterate %d,  |step| = %.3g', whichIt(q), dnorm));
    if q == 1, legend(ax, 'Location', 'southwest', 'FontSize', 8); end
end
saveFig(ctx, fig, 'step_spectrum');

%% 5. What the iterates look like
% Source deformed by the iterates 0, 1, 2 and the last one, over the
% target. Axes limits are fixed to the extent of the two input meshes;
% parts that left that box are clipped.
show = [1 2 3 nIt + 1];
bnd = meshBounds(model.src.V, model.tar.V);
fig = newFig(ctx, 8, 5.2);
axs = panelGrid(fig, 1, 4, 'Margins', [0.005 0.005 0.08 0.11], 'Gap', [0.005 0]);
for q = 1:4
    Vq = edsm.deform(gnTrace(:, show(q)), model);
    drawOverlay(axs(q), Vq, model.src.F, model.tar.V, model.tar.F);
    setBodyView(axs(q), bnd);
    panelTitle(axs(q), sprintf('iteration %d\nerr %.3g', show(q) - 1, hist.err(show(q))), 'FontSize', 10);
end
annotation(fig, 'textbox', [0 0 1 0.06], 'String', 'blue = deformed source, orange = target', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11);
saveFig(ctx, fig, 'iterates');

tutorialDone(ctx);

%% Local function
function putIterate(store, it, P)
store(it) = P(:);
end
