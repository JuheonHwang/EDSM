%% t00_quickstart  Register one mesh onto another in about a minute
% Loads the source/target mesh pair, builds the Embedded Deformation (ED)
% model and runs Levenberg-Marquardt (the default solver). Shows the meshes
% before and after, and how the error went down.
%
% Run:  setup; run('tutorials/t00_quickstart.m')
%       matlab -batch "cd('<repo>'); setup; run('tutorials/t00_quickstart.m')"
clearvars; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'helpers'));
ctx = tutorialInit('t00_quickstart');
col = tutorialColors();

%% 1. Load the meshes and build the model
% buildModel precomputes everything that does not depend on the parameters:
% which vertices become ED nodes, the vertex-to-node blending weights, the
% node-to-node neighbor graph and the mesh Laplacian. initParam returns the
% identity deformation (A_i = I, t_i = 0, R = I, T = 0) as the starting point.
[model, P0] = loadTutorialModel(ctx);

%% 2. Residual at the starting point
% The objective is err(P) = sum(F(P).^2), where F is the residual vector of
% edsm.residual. err(P0) is the cost of leaving the source mesh untouched.
F0 = edsm.residual(P0, model);
fprintf('Residual vector: %d entries, err(P0) = sum(F.^2) = %.4f\n', numel(F0), sum(F0.^2));

%% 3. Run Levenberg-Marquardt
% 30 accepted iterations; opts.verbose prints one line per iteration. The
% solver returns the final parameters P and a history struct (error per
% iteration, damping lambda, accepted/rejected steps, cumulative time).
opts = struct('maxIter', 30, 'verbose', true);
[P, hist] = edsm.solvers.levenbergMarquardt(P0, model, opts);
fprintf('\nLM finished: %d iterations, stop reason "%s", %.1f s\n', hist.nIter, hist.stopReason, hist.time(end));
fprintf('err: %.4f -> %.4f  (%.1f%% of the start value)\n', hist.err(1), hist.err(end), 100 * hist.err(end) / hist.err(1));

%% 4. Deform the source and measure the fit in mesh units
% edsm.deform applies the parameters to the source vertices. For every
% target vertex we take the distance to the nearest deformed-source vertex.
Vdef = edsm.deform(P, model);
d0 = fitDistance(model.src.V, model.tar.V);
d1 = fitDistance(Vdef, model.tar.V);
fprintf('Distance target vertex -> nearest source vertex (mesh units):\n');
fprintf('  before: mean %.2f  median %.2f  95%% %.2f  max %.2f\n', mean(d0), median(d0), pct(d0, 95), max(d0));
fprintf('  after : mean %.2f  median %.2f  95%% %.2f  max %.2f\n', mean(d1), median(d1), pct(d1, 95), max(d1));

%% 5. Figure: before / after, source (blue) over target (orange)
% Both surfaces are opaque: the one in front wins the depth test. A patchwork
% of blue and orange means the surfaces coincide; a large single-color region
% means they are apart there. Front view, head up.
fig = newFig(ctx, 8, 5.4);
axs = panelGrid(fig, 1, 2, 'Margins', [0.01 0.01 0.08 0.09], 'Gap', [0.02 0]);
bnd = meshBounds(model.src.V, model.tar.V, Vdef);
titles = {sprintf('Before  (err = %.1f)', hist.err(1)), sprintf('After %d LM iterations  (err = %.1f)', hist.nIter, hist.err(end))};
Vs = {model.src.V, Vdef};
for k = 1:2
    drawOverlay(axs(k), Vs{k}, model.src.F, model.tar.V, model.tar.F);
    setBodyView(axs(k), bnd);
    panelTitle(axs(k), titles{k});
end
annotation(fig, 'textbox', [0 0 1 0.06], 'String', 'blue = deformed source     orange = target', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', col.text);
saveFig(ctx, fig, 'before_after');

%% 6. Figure: convergence
% Error per accepted iteration and the damping lambda that
% produced each step.
fig = newFig(ctx, 8, 3.6);
subplot(1, 2, 1);
plot(0:hist.nIter, hist.err, '-o', 'Color', col.lm, 'MarkerFaceColor', col.lm, 'MarkerSize', 4);
xlabel('LM iteration'); ylabel('error  \Sigma F_k^2'); ylim([140 290]); title('Objective'); grid on;
subplot(1, 2, 2);
semilogy(1:hist.nIter, hist.lambda, '-o', 'Color', col.lmEye, 'MarkerFaceColor', col.lmEye, 'MarkerSize', 4);
xlabel('LM iteration'); ylabel('damping \lambda'); title('Damping of the accepted step'); grid on;
saveFig(ctx, fig, 'convergence');

tutorialDone(ctx);
