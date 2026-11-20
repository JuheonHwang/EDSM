function d = deformCore(R0, T0, A0, t0, model)
%DEFORMCORE  Embedded-deformation of the source vertices (shared by deform/residual).
%
%   d = deformCore(R0, T0, A0, t0, model)
%
%   Inputs: decoded parameters (see edsm.decodeParam) and the model.
%   Output (struct), with n = nVerts, K = nVertexNeighbor:
%     V            : n x 3  deformed vertices
%     local        : n x 3  local transform, sum_k w_k (A_k (v - g_k) + g_k + t_k) - G0
%     v1           : 3 x 1 x n x K   v - g_k  (before applying A_k)
%     distNeighbor_: 1 x n x K       blending weights, permuted for broadcasting
%     idxNeighbor  : K x n           node indices, NaN replaced by 1
%     invalid      : 1 x n logical   vertices flagged by NaN neighbor indices
%     v0           : 3 x n           source vertices (transposed)
%     g0           : 3 x nNodes      node positions
%
%   Same expressions, in the same order, as EfuncLocal / deformED of the
%   original scripts.

v0 = model.src.V';                     % 3 x n
g0 = v0(:, model.nodeIdx);             % 3 x nNodes
G0 = model.G0;
idxNeighbor = model.idxNeighbor;       % K x n
distNeighbor = model.distNeighbor;     % K x n

n_verts = size(distNeighbor, 2);
n_neighbor = size(distNeighbor, 1);
assert(n_verts > n_neighbor);

distNeighbor_ = reshape(distNeighbor', [1 size(distNeighbor')]);   % 1 x n x K
invalidNeibor = isnan(idxNeighbor);
idxNeighbor(invalidNeibor) = 1;

g1 = reshape(g0(:, idxNeighbor'), [size(g0, 1) size(idxNeighbor')]);                 % 3 x n x K
t1 = reshape(t0(:, idxNeighbor'), [size(t0, 1) size(idxNeighbor')]);                 % 3 x n x K
A1 = reshape(A0(:, :, idxNeighbor'), [size(A0, [1 2]) size(idxNeighbor')]);          % 3 x 3 x n x K
v1 = reshape(v0 - g1, [size(g1, 1) 1 size(g1, 2:3)]);                                % 3 x 1 x n x K
v1_trans = squeeze(pagemtimes(A1, v1)) + g1 + t1;                                    % 3 x n x K

deformedV_local = squeeze(sum(distNeighbor_ .* v1_trans, 3))' - G0(:)';   % Local Transform, n x 3
deformedV = deformedV_local * R0' + T0(:)' + G0(:)';                      % Global Transform, n x 3
% Vertices without valid neighbors keep the source position (independent of P);
% edsm.residual zeroes their Jacobian rows unless cfg.legacyJacobian is set.
deformedV(any(invalidNeibor, 1), :) = v0(:, any(invalidNeibor, 1))';

d.V = deformedV;
d.local = deformedV_local;
d.v1 = v1;
d.distNeighbor_ = distNeighbor_;
d.idxNeighbor = idxNeighbor;
d.invalid = any(invalidNeibor, 1);
d.v0 = v0;
d.g0 = g0;
end
