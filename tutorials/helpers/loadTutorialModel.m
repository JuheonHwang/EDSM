function [model, P0] = loadTutorialModel(ctx, cfgOverrides)
%LOADTUTORIALMODEL  Load the default mesh pair and build the ED model.
%
%   [model, P0] = loadTutorialModel(ctx)
%   [model, P0] = loadTutorialModel(ctx, struct('legacyJacobian', true))
%
%   Fields of cfgOverrides replace those of edsm.defaultConfig(). Prints the
%   problem size.

cfg = edsm.defaultConfig();
if nargin > 1 && ~isempty(cfgOverrides)
    f = fieldnames(cfgOverrides);
    for k = 1:numel(f), cfg.(f{k}) = cfgOverrides.(f{k}); end
end
t = tic;
src = edsm.loadMesh(fullfile(ctx.repoRoot, 'data', 'simple_poisson_mesh_020.ply'));
tar = edsm.loadMesh(fullfile(ctx.repoRoot, 'data', 'simple_poisson_mesh_022.ply'));
model = edsm.buildModel(src, tar, cfg);
P0 = edsm.initParam(model);
fprintf('Model built in %.1f s: source %d vertices / %d faces, target %d vertices, %d ED nodes, %d parameters (legacyJacobian=%d)\n', ...
    toc(t), model.nVerts, size(src.F, 1), size(tar.V, 1), model.nNodes, numel(P0), cfg.legacyJacobian);
end
