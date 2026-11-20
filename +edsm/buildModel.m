function model = buildModel(src, tar, cfg)
%BUILDMODEL  Precompute everything that does not depend on the parameters P.
%
%   model = edsm.buildModel(src, tar)
%   model = edsm.buildModel(src, tar, cfg)
%
%   Inputs:
%     src, tar : meshes from edsm.loadMesh (fields V, F, N)
%     cfg      : struct from edsm.defaultConfig (default if omitted)
%   Output (struct):
%     cfg           : the configuration used
%     src, tar      : the input meshes (V n x 3, F m x 3, N n x 3), plain double
%     nVerts        : number of source vertices
%     nNodes        : number of ED nodes, round(nVerts * cfg.sampleRate)
%     nodeIdx       : 1 x nNodes, source vertex index of each ED node
%                     (rng(seed,'twister'); randperm(nVerts, nNodes))
%     g0            : 3 x nNodes, node positions src.V(nodeIdx,:)'
%     G0            : 1 x 3, centroid of the source vertices
%     idxNeighbor   : nVertexNeighbor x nVerts, ED nodes influencing each vertex
%                     (NaN column = vertex without valid neighbors)
%     distNeighbor  : nVertexNeighbor x nVerts, normalized blending weights
%     idxNodeNeighbor : nNodeNeighbor x nNodes, neighbors of each node
%                     (first entry is the node itself, in both distance modes)
%     Lap           : nVerts x nVerts sparse graph Laplacian (geodesic edge weights)
%     lapDiag       : nVerts x 1, full(diag(Lap))
%
%   The global random generator state is restored on return.
%   Replicates the setup section of the EDRegisterSparse_Show_iter_* scripts.

if nargin < 3
    cfg = edsm.defaultConfig();
end
nVertexNeighbor = cfg.nVertexNeighbor;
nNodeNeighbor = cfg.nNodeNeighbor;
assert(nVertexNeighbor > 1);
assert(nNodeNeighbor > 1);

srcV = src.V;
srcF = src.F;
nVerts = size(srcV, 1);
nNodes = round(nVerts * cfg.sampleRate);

% ---- node sampling (same as the originals) -------------------------------
rngState = rng;
rng(cfg.seed, 'twister');
nodeIdx = randperm(nVerts, nNodes);
rng(rngState);

G0 = mean(srcV, 1);

% ---- vertex -> node weights, node -> node neighbors, Laplacian -----------
s = [srcF(:, 1); srcF(:, 2); srcF(:, 3)];
t = [srcF(:, 2); srcF(:, 3); srcF(:, 1)];
w = sqrt(sum((srcV(s, :) - srcV(t, :)).^2, 2));
G = graph(s, t, w);
G = simplify(G);
% NOTE(review): in the originals Lap is only defined inside the geodesic branch,
% so distanceOnMesh = false crashed at its first use. Here Lap is built in both
% modes (same expression), which is the only change in that branch.
Lap = laplacian(G);

if cfg.distanceOnMesh
    D = distances(G);                     % nVerts x nVerts (dense)
    D = D(:, nodeIdx);                    % nVerts x nNodes
    [distNeighbor, idxNeighbor] = mink(D, nVertexNeighbor + 1, 2);
    distNeighbor = distNeighbor'; idxNeighbor = idxNeighbor';   % (K+1) x nVerts
    D = D(nodeIdx, :);                    % nNodes x nNodes
    [distNodeNeighbor, idxNodeNeighbor] = mink(D, nNodeNeighbor, 2);
    idxNodeNeighbor = idxNodeNeighbor';   % nNodeNeighbor x nNodes
    % NOTE(review): isnan never fires for disconnected nodes (distance Inf), so
    % this assert does not guard what it seems to guard.
    assert(~any(isnan(distNodeNeighbor(:))));

    % Weight = 1 - d/d_max, d_max = distance to the (K+1)-th nearest node.
    distNeighbor = 1.0 - (distNeighbor ./ distNeighbor(nVertexNeighbor + 1, :));
    idxNeighbor = idxNeighbor(1:nVertexNeighbor, :); distNeighbor = distNeighbor(1:nVertexNeighbor, :);
else
    nodeV = srcV(nodeIdx, :);
    [idxNeighbor, distNeighbor] = knnBrute(nodeV, srcV, nVertexNeighbor + 1);     % (K+1) x nVerts, squared distances
    idxNodeNeighbor = knnBrute(nodeV, nodeV, nNodeNeighbor);                      % nNodeNeighbor x nNodes, self first
    distNeighbor = sqrt(distNeighbor);

    % Same as the geodesic branch: weight = 1 - d/d_max, d_max = (K+1)-th distance.
    distNeighbor = 1.0 - (distNeighbor ./ distNeighbor(nVertexNeighbor + 1, :));
    idxNeighbor = idxNeighbor(1:nVertexNeighbor, :); distNeighbor = distNeighbor(1:nVertexNeighbor, :);
end

distNeighbor = distNeighbor ./ sum(distNeighbor, 1);   % Normalize Dist Weights

invalidIdx = any(isnan(distNeighbor), 1);
% Fill value 1 / nNodeNeighbor is kept in both Jacobian modes (legacy parity). It
% only enters the legacy Jacobian: F overwrites these vertices with the source
% position and the corrected Jacobian zeroes their derivatives.
distNeighbor(:, invalidIdx) = 1 / nNodeNeighbor;
idxNeighbor(:, invalidIdx) = NaN;

model.cfg = cfg;
model.src = struct('V', double(src.V), 'F', src.F, 'N', src.N);
model.tar = struct('V', double(tar.V), 'F', tar.F, 'N', tar.N);
model.nVerts = nVerts;
model.nNodes = nNodes;
model.nodeIdx = nodeIdx;
model.g0 = srcV(nodeIdx, :)';
model.G0 = G0;
model.idxNeighbor = idxNeighbor;
model.distNeighbor = distNeighbor;
model.idxNodeNeighbor = idxNodeNeighbor;
model.Lap = Lap;
model.lapDiag = full(diag(Lap));
end
