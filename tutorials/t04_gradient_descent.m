%% t04_gradient_descent  Fixed-step gradient descent on the ED objective
% The simplest optimizer: P <- P - alpha * grad, with grad = 2 J'F the
% gradient of err = sum(F.^2). edsm.solvers.gradientDescent uses one step
% size alpha (stepSize) for the 600 local parameters A_i, t_i and a smaller
% one, alpha*globalStepScale, for the 6 global parameters r, T. There is no
% line search and no convergence test: it just runs maxIter iterations.
clearvars; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'helpers'));
ctx = tutorialInit('t04_gradient_descent');
col = tutorialColors();

%% 1. The gradient at the start and what limits the step size
% For a quadratic model the iteration is stable only if
% alpha * 2 * lambda_max < 2, where lambda_max is the largest eigenvalue of
% J'J (scaled by the per-parameter step factors). The gradient is dominated
% by the global rotation r, which is why r and T get their own smaller step.
[model, P0] = loadTutorialModel(ctx);
[F0, J0] = edsm.residual(P0, model);
g0 = 2 * (J0' * F0);
n = model.nNodes;
fprintf('err(P0) = %.4f, |grad| = %.4g\n', sum(F0.^2), norm(g0));
fprintf('gradient norm by parameter group: r %.4g | T %.4g | A %.4g | t %.4g\n', norm(g0(1:3)), norm(g0(4:6)), norm(g0(7:6 + 9 * n)), norm(g0(7 + 9 * n:end)));
H = full(J0' * J0);
lamAll = max(eig((H + H') / 2));
fprintf('lambda_max(J''J) = %.4g\n', lamAll);
fprintf('Stability bound at P0, alpha_max = 1/lambda_max(S^(1/2) J''J S^(1/2)), S = diag of step factors:\n');
for gs = [0 0.01 0.1 1]
    s = [gs * ones(6, 1); ones(numel(P0) - 6, 1)];
    Hs = sqrt(s) .* H .* sqrt(s)';
    l = max(eig((Hs + Hs') / 2));
    fprintf('  globalStepScale %-5g : lambda_max = %10.4g   alpha_max = %.3g\n', gs, l, 1 / l);
end
fprintf('(The bound is computed at P0 only; J changes along the path.)\n');

%% 2. The default run
% stepSize 1e-3, globalStepScale 0.01, 100 iterations.
[Pd, hd] = edsm.solvers.gradientDescent(P0, model, struct('maxIter', 100));
[em, km] = min(hd.err);
fprintf('\ndefault run: err %.4f -> minimum %.4f at iteration %d -> final %.4f after %d iterations (%.1f s)\n', ...
    hd.err(1), em, km - 1, hd.err(end), hd.nIter, hd.time(end));
inc = find(diff(hd.err) > 0);
fprintf('iterations with an INCREASE of the error: %d of %d (first at iteration %d)\n', numel(inc), hd.nIter, inc(1));
fprintf('iter  err        |step|\n');
for k = [0 1 2 5 10 20 40 60 80 100]
    if k == 0, fprintf('%4d  %9.4f\n', k, hd.err(1)); else, fprintf('%4d  %9.4f  %.3g\n', k, hd.err(k + 1), hd.stepNorm(k)); end
end
fprintf('|grad| at the final iterate = %.4g (it was %.4g at P0)\n', norm(gradAt(Pd, model)), norm(g0));

fig = newFig(ctx, 8, 3.8);
ax = axes(fig, 'Position', [0.115 0.17 0.37 0.74]);
plot(ax, 0:hd.nIter, hd.err, '-', 'Color', col.gd); hold(ax, 'on');
plot(ax, km - 1, em, 'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
text(ax, km - 1 + 4, em + 110, sprintf('min %.1f at iter %d', em, km - 1), 'FontSize', 10);
xlabel(ax, 'iteration'); ylabel(ax, 'error  \Sigma F_k^2'); title(ax, 'Error (default: \alpha = 10^{-3})');
ax = axes(fig, 'Position', [0.58 0.17 0.39 0.74]);
semilogy(ax, 1:hd.nIter, hd.stepNorm, '-', 'Color', col.gd);
xlabel(ax, 'iteration'); ylabel(ax, '||step||_2'); title(ax, 'Step length');
saveFig(ctx, fig, 'default_run');

%% 3. Effect of the step size
% Same start, 100 iterations, stepSize from far too small to too large.
% Compare with alpha_max of section 1 (2.5e-4 for the default globalStepScale):
% 3e-3 diverges to Inf, 1e-3 is neither stable nor divergent.
steps = [1e-5 1e-4 3e-4 1e-3 3e-3];
H1 = cell(size(steps));
fprintf('\nstep size sweep (globalStepScale 0.01):\n');
fprintf('  %-8s %6s %12s %12s %8s %-10s\n', 'alpha', 'iters', 'final err', 'min err', 'argmin', 'stop');
for k = 1:numel(steps)
    if steps(k) == 1e-3
        H1{k} = hd;
    else
        [~, H1{k}] = edsm.solvers.gradientDescent(P0, model, struct('maxIter', 100, 'stepSize', steps(k)));
    end
    [mn, am] = min(H1{k}.err);
    fprintf('  %-8.0e %6d %12.5g %12.5g %8d %-10s\n', steps(k), H1{k}.nIter, H1{k}.err(end), mn, am - 1, H1{k}.stopReason);
end
fig = sweepFigure(ctx, H1, arrayfun(@(a) sprintf('\\alpha = %.0e', a), steps, 'UniformOutput', false), find(steps == 1e-3), col);
saveFig(ctx, fig, 'step_size');

%% 4. Effect of globalStepScale
% Keep stepSize = 1e-3 and change the factor for r and T. The gradient on r
% is two orders of magnitude larger than on the rest (section 1), so the
% global parameters need the small factor.
scales = [0 0.01 0.1 1];
H2 = cell(size(scales));
fprintf('\nglobalStepScale sweep (stepSize 1e-3):\n');
fprintf('  %-8s %6s %12s %12s %8s %-10s\n', 'scale', 'iters', 'final err', 'min err', 'argmin', 'stop');
for k = 1:numel(scales)
    if scales(k) == 0.01
        H2{k} = hd;
    else
        [~, H2{k}] = edsm.solvers.gradientDescent(P0, model, struct('maxIter', 100, 'globalStepScale', scales(k)));
    end
    [mn, am] = min(H2{k}.err);
    fprintf('  %-8g %6d %12.5g %12.5g %8d %-10s\n', scales(k), H2{k}.nIter, H2{k}.err(end), mn, am - 1, H2{k}.stopReason);
end
fig = sweepFigure(ctx, H2, arrayfun(@(a) sprintf('globalStepScale = %g', a), scales, 'UniformOutput', false), find(scales == 0.01), col);
saveFig(ctx, fig, 'global_step_scale');

tutorialDone(ctx);

%% Local functions
function g = gradAt(P, model)
[F, J] = edsm.residual(P, model);
g = 2 * (J' * F);
end

function fig = sweepFigure(ctx, Hs, labels, iDefault, col)
% Error curves of several runs on one axes; diverged runs leave the chart.
fig = newFig(ctx, 8, 4.4);
ax = axes(fig, 'Position', [0.09 0.15 0.88 0.78]);
cm = viridisMap(numel(Hs) + 2); cm = cm(2:end - 1, :);
hold(ax, 'on');
hl = gobjects(1, numel(Hs));
for k = 1:numel(Hs)
    e = Hs{k}.err;
    c = cm(k, :); lw = 1.6;
    if k == iDefault, c = col.gd; lw = 2.6; end
    lab = labels{k};
    if strcmp(Hs{k}.stopReason, 'nonFinite')
        lab = sprintf('%s  (non-finite at iteration %d)', lab, Hs{k}.nIter);
    end
    hl(k) = semilogy(ax, 0:numel(e) - 1, e, '-', 'Color', c, 'LineWidth', lw, 'DisplayName', lab);
end
set(ax, 'YScale', 'log');
yline(ax, Hs{1}.err(1), ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
ylim(ax, [100 1e4]); xlim(ax, [0 100]);
xlabel(ax, 'iteration'); ylabel(ax, 'error  \Sigma F_k^2  (clipped to [10^2, 10^4])');
legend(ax, hl, 'Location', 'northeast', 'FontSize', 9);
title(ax, 'Error vs iteration (orange = default); dotted: error at P0');
end
