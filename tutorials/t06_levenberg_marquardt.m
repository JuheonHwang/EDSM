%% t06_levenberg_marquardt  Levenberg-Marquardt: damping between GN and gradient descent
% LM solves (J'J + lambda*D) d = -J'F with a damping matrix D (diag(J'J) or
% the identity). Small lambda gives the Gauss-Newton step, large lambda a
% short step along the (scaled) negative gradient. The solver adapts lambda:
% a step that reduces the error is accepted and lambda divided by 10, a step
% that does not is rejected and lambda multiplied by 10.
clearvars; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'helpers'));
ctx = tutorialInit('t06_levenberg_marquardt');
col = tutorialColors();

%% 1. lambda as a bridge: the damped step at P0 for many lambdas
% For each lambda solve the damped normal equations at P0 and compare the
% step d(lambda) with the Gauss-Newton step (lsqminnorm) and with the
% steepest-descent direction -g. Angles are in degrees.
[model, P0] = loadTutorialModel(ctx);
[F0, J0] = edsm.residual(P0, model);
JtJ = full(J0' * J0); g0 = J0' * F0; err0 = sum(F0.^2);
dGN = lsqminnorm(J0, -F0);
dg = diag(JtJ);
fprintf('diag(J''J): min %.3e, max %.3e, number of zeros %d\n', min(dg), max(dg), sum(dg == 0));
ang = @(a, b) acosd(max(-1, min(1, (a' * b) / (norm(a) * norm(b)))));
lams = logspace(-6, 6, 25);
damp = {'diag', 'eye'};
Dmat = {diag(dg), eye(numel(dg))};
nl = numel(lams);
aGN = zeros(nl, 2); aGD = zeros(nl, 2); sn = zeros(nl, 2); ea = zeros(nl, 2); ep = zeros(nl, 2);
for m = 1:2
    for i = 1:nl
        d = (JtJ + lams(i) * Dmat{m}) \ (-g0);
        aGN(i, m) = ang(d, dGN);
        aGD(i, m) = ang(d, -Dmat{m} \ g0);       % direction of the damping-scaled negative gradient
        sn(i, m) = norm(d);
        ep(i, m) = sum((F0 + J0 * d).^2);
        ea(i, m) = sum(edsm.residual(P0 + d, model).^2);
    end
end
fprintf('\nlambda      | diag: angle to GN  |step|   err(P0+d) | eye: angle to GN  angle to -g  |step|   err(P0+d)\n');
for i = 1:2:nl
    fprintf('%9.2e   | %14.2f  %9.4g  %9.4g | %13.2f  %11.2f  %9.4g  %9.4g\n', lams(i), aGN(i, 1), sn(i, 1), ea(i, 1), aGN(i, 2), ang(-g0, (JtJ + lams(i) * eye(numel(dg))) \ (-g0)), sn(i, 2), ea(i, 2));
end
fprintf('GN step: |d| = %.4g, err(P0 + dGN) = %.4g;  err(P0) = %.4g\n', norm(dGN), sum(edsm.residual(P0 + dGN, model).^2), err0);
[~, ib] = min(ea(:, 1));
fprintf('best lambda on this grid (diag damping): %.2e with err %.4g\n', lams(ib), ea(ib, 1));

fig = newFig(ctx, 8, 4.6);
ax = axes(fig, 'Position', [0.07 0.14 0.28 0.78]);
semilogx(ax, lams, aGN(:, 1), '-', 'Color', col.lm, 'DisplayName', 'diag: angle to GN step'); hold(ax, 'on');
semilogx(ax, lams, aGD(:, 1), '--', 'Color', col.lm, 'DisplayName', 'diag: angle to -D^{-1}g');
semilogx(ax, lams, aGN(:, 2), '-', 'Color', col.lmEye, 'DisplayName', 'eye: angle to GN step');
semilogx(ax, lams, aGD(:, 2), '--', 'Color', col.lmEye, 'DisplayName', 'eye: angle to -g');
xlabel(ax, '\lambda'); ylabel(ax, 'angle (degrees)'); title(ax, 'Direction'); xlim(ax, [1e-6 1e6]); xticks(ax, 10.^(-6:4:6));
legend(ax, 'Location', 'south', 'FontSize', 7);
ax = axes(fig, 'Position', [0.42 0.14 0.28 0.78]);
loglog(ax, lams, sn(:, 1), '-', 'Color', col.lm, 'DisplayName', 'diag'); hold(ax, 'on');
loglog(ax, lams, sn(:, 2), '-', 'Color', col.lmEye, 'DisplayName', 'eye');
yline(ax, norm(dGN), ':', 'Color', col.gn, 'DisplayName', '|GN step|');
xlabel(ax, '\lambda'); ylabel(ax, '||d||_2'); title(ax, 'Step length'); xlim(ax, [1e-6 1e6]); xticks(ax, 10.^(-6:4:6));
legend(ax, 'Location', 'southwest', 'FontSize', 8);
ax = axes(fig, 'Position', [0.77 0.14 0.21 0.78]);
loglog(ax, lams, ea(:, 1), '-', 'Color', col.lm); hold(ax, 'on');
loglog(ax, lams, ea(:, 2), '-', 'Color', col.lmEye);
yline(ax, err0, ':', 'Color', 'k'); 
text(ax, 2e-6, err0 * 1.05, 'err(P0)', 'FontSize', 8);
xlabel(ax, '\lambda'); ylabel(ax, 'err(P_0 + d)'); title(ax, 'Actual error'); xlim(ax, [1e-6 1e6]); xticks(ax, 10.^(-6:6:6));
ylim(ax, [150 1e3]); yticks(ax, [200 300 500 1000]);
saveFig(ctx, fig, 'lambda_bridge');

%% 2. A full LM run (default settings): lambda, accepts and rejects
% The history holds, per accepted iteration, the lambda that produced the
% step (lambda), the lambda carried into the next iteration (lambdaNext) and
% how many trial steps were rejected first (nReject).
[Pd, hd] = edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 100));
hasGrad = isfield(hd, 'gradNorm');
fprintf('\nLM default (damping diag, direct): %d accepted iterations, stop reason "%s", %.1f s\n', hd.nIter, hd.stopReason, hd.time(end));
fprintf('err: %.4f -> %.4f;  hist.gradNorm available: %d\n', hd.err(1), hd.err(end), hasGrad);
fprintf('rejected trial steps in total: %d; accepted iterations that needed rejections: %d of %d\n', sum(hd.nReject), sum(hd.nReject > 0), hd.nIter);
fprintf('lambda used: min %.2e, max %.2e\n', min(hd.lambda), max(hd.lambda));
fprintf('iter  err         lambda    nReject  lambdaNext  |step|\n');
for k = unique([1:12, hd.nIter])
    fprintf('%4d  %-10.5f  %-8.1e  %-7d  %-10.1e  %.4g\n', k, hd.err(k + 1), hd.lambda(k), hd.nReject(k), hd.lambdaNext(k), hd.stepNorm(k));
end
if hasGrad
    fprintf('gradient inf-norm ||2 J''F||_inf (hist.gradNorm): start %.4g, final %.4g\n', hd.gradNorm(1), hd.gradNorm(end));
end

nP = 3 + hasGrad;
fig = newFig(ctx, 8, 2.1 * nP);
ax = gobjects(nP, 1);
for q = 1:nP
    ax(q) = axes(fig, 'Position', [0.10, 0.08 + (nP - q) * 0.88 / nP, 0.87, 0.88 / nP - 0.045]);
end
semilogy(ax(1), 0:hd.nIter, hd.err, '-o', 'Color', col.lm, 'MarkerFaceColor', col.lm, 'MarkerSize', 3);
ylabel(ax(1), 'error  \Sigma F_k^2'); ylim(ax(1), [145 290]); yticks(ax(1), [150 200 250]); title(ax(1), 'LM, diag damping, direct solver');
semilogy(ax(2), 1:hd.nIter, hd.lambda, '-o', 'Color', col.lmEye, 'MarkerFaceColor', col.lmEye, 'MarkerSize', 3);
ylabel(ax(2), '\lambda of the step'); ylim(ax(2), [1e-9 1e-1]); yticks(ax(2), 10.^(-8:2:-2));
bar(ax(3), 1:hd.nIter, hd.nReject, 1, 'FaceColor', col.gd, 'EdgeColor', 'none');
ylabel(ax(3), '# rejected trials'); ax(3).YLim = [0 max(1, max(hd.nReject)) + 0.5]; ax(3).YTick = 0:max(1, max(hd.nReject));
if hasGrad
    semilogy(ax(4), 0:hd.nIter, hd.gradNorm, '-o', 'Color', col.lmCG, 'MarkerFaceColor', col.lmCG, 'MarkerSize', 3);
    ylabel(ax(4), '||2J^TF||_{\infty}');
end
for q = 1:nP, xlim(ax(q), [0 hd.nIter + 1]); if q < nP, ax(q).XTickLabel = []; end, end
xlabel(ax(nP), 'accepted iteration');
saveFig(ctx, fig, 'history');

%% 3. Damping matrix: diag(J'J) vs identity
% 'diag' scales the damping per parameter (Marquardt), 'eye' damps all
% parameters equally (Levenberg). The parameters have very different scales
% (rotation vector, matrices near I, translations of the order of the mesh).
[Pe, he] = edsm.solvers.levenbergMarquardt(P0, model, struct('maxIter', 100, 'damping', 'eye'));
fprintf('\nLM damping eye : %d iterations, stop "%s", err %.4f, %.1f s, rejected trials %d\n', he.nIter, he.stopReason, he.err(end), he.time(end), sum(he.nReject));
fprintf('LM damping diag: %d iterations, stop "%s", err %.4f, %.1f s, rejected trials %d\n', hd.nIter, hd.stopReason, hd.err(end), hd.time(end), sum(hd.nReject));
fprintf('error after 10 / 20 / 40 iterations: diag %.3f / %.3f / %.3f ; eye %.3f / %.3f / %.3f\n', ...
    hd.err(min(11, end)), hd.err(min(21, end)), hd.err(min(41, end)), he.err(min(11, end)), he.err(min(21, end)), he.err(min(41, end)));
fig = newFig(ctx, 8, 3.8);
ax = axes(fig, 'Position', [0.09 0.17 0.40 0.74]);
plot(ax, 0:hd.nIter, hd.err, '-', 'Color', col.lm, 'DisplayName', 'diag'); hold(ax, 'on');
plot(ax, 0:he.nIter, he.err, '-', 'Color', col.lmEye, 'DisplayName', 'eye');
set(ax, 'YScale', 'log'); ylim(ax, [140 300]);
xlabel(ax, 'accepted iteration'); ylabel(ax, 'error  \Sigma F_k^2'); title(ax, 'Error'); legend(ax, 'Location', 'northeast');
ax = axes(fig, 'Position', [0.58 0.17 0.39 0.74]);
semilogy(ax, 1:hd.nIter, hd.lambda, '-', 'Color', col.lm, 'DisplayName', 'diag'); hold(ax, 'on');
semilogy(ax, 1:he.nIter, he.lambda, '-', 'Color', col.lmEye, 'DisplayName', 'eye');
xlabel(ax, 'accepted iteration'); ylabel(ax, '\lambda of the step'); title(ax, 'Damping'); legend(ax, 'Location', 'southeast');
saveFig(ctx, fig, 'damping');

%% 4. Corrected vs legacy Jacobian
% Same solver, same start, but model.cfg.legacyJacobian = true, which uses
% the Jacobian of the original scripts (known errors, see t03). F is
% identical, so the error values are comparable; the gradient seen by the
% solver is not. The table gives ||2 J'F|| at the final points computed with
% the legacy J (what a solver using it believes) and with the corrected J.
[modelL, ~] = loadTutorialModel(ctx, struct('legacyJacobian', true));
[Pl, hl] = edsm.solvers.levenbergMarquardt(P0, modelL, struct('maxIter', 100));
fprintf('\nLM, legacy Jacobian: %d iterations, stop "%s", err %.4f, %.1f s\n', hl.nIter, hl.stopReason, hl.err(end), hl.time(end));
gn = @(P, mdl) norm(2 * (J_of(P, mdl)' * edsm.residual(P, mdl)));
tab = [gn(Pl, modelL), gn(Pl, model); gn(Pd, modelL), gn(Pd, model)];
fprintf('\n||2 J''F|| at the final point of each run:\n');
fprintf('  %-28s %16s %18s\n', 'run', 'legacy J', 'corrected J');
fprintf('  %-28s %16.4g %18.4g\n', 'LM with legacy Jacobian', tab(1, :));
fprintf('  %-28s %16.4g %18.4g\n', 'LM with corrected Jacobian', tab(2, :));
fprintf('err at the final points: legacy-J run %.4f, corrected-J run %.4f (difference %.2f)\n', hl.err(end), hd.err(end), hl.err(end) - hd.err(end));
if isfield(hl, 'gradNorm')
    fprintf('hist.gradNorm (inf-norm) final: legacy run %.4g, corrected run %.4g\n', hl.gradNorm(end), hd.gradNorm(end));
end

fig = newFig(ctx, 8, 3.8);
ax = axes(fig, 'Position', [0.09 0.17 0.42 0.74]);
plot(ax, 0:hd.nIter, hd.err, '-', 'Color', col.lm, 'DisplayName', 'corrected Jacobian'); hold(ax, 'on');
plot(ax, 0:hl.nIter, hl.err, '--', 'Color', col.lm, 'LineWidth', 2.2, 'DisplayName', 'legacy Jacobian');
set(ax, 'YScale', 'log'); ylim(ax, [140 300]);
xlabel(ax, 'accepted iteration'); ylabel(ax, 'error  \Sigma F_k^2');
title(ax, 'Same solver, different Jacobian'); legend(ax, 'Location', 'northeast');
text(ax, hl.nIter, hl.err(end) * 1.03, sprintf('stops at %.1f', hl.err(end)), 'FontSize', 9, 'HorizontalAlignment', 'right');
ax = axes(fig, 'Position', [0.62 0.17 0.35 0.74]);
b = bar(ax, tab, 'grouped');
b(1).FaceColor = col.legacy; b(2).FaceColor = col.lm;
set(ax, 'YScale', 'log', 'XTickLabel', {'run with legacy J', 'run with corrected J'});
ylabel(ax, '||2 J^TF||_2 at the final point'); title(ax, 'Final gradient norm');
legend(ax, {'measured with legacy J', 'measured with corrected J'}, 'Location', 'northeast', 'FontSize', 8);
ylim(ax, [1e-8 1e8]);
saveFig(ctx, fig, 'jacobian_modes');

tutorialDone(ctx);

%% Local function
function J = J_of(P, mdl)
[~, J] = edsm.residual(P, mdl);
end
