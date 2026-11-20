function [F, J, info] = residual(P, model, indices)
%RESIDUAL  Residual vector and sparse analytic Jacobian of the ED registration.
%
%   F = edsm.residual(P, model)
%   [F, J] = edsm.residual(P, model)
%   [F, J, info] = edsm.residual(P, model, indices)
%
%   Inputs:
%     P       : (6 + 12n) parameter vector [r(3); T(3); A(:); t(:)]
%     model   : struct from edsm.buildModel
%     indices : (optional) nTar x 1 nearest-neighbor correspondences, row of
%               the deformed source for each target vertex. Computed with
%               edsm.nearestIdx when omitted. Pass them to hold the
%               correspondences fixed (e.g. for finite differences).
%   Outputs:
%     F    : residual column,
%            [Efit(:); wRot*Erot1(:); wRot*Erot2(:); wSmooth*Esmooth(:); wLap*Elap(:)]
%            sizes 3*nTar, 9n, n, 3n, 3*nVerts (n = nNodes)
%     J    : sparse (numel(F) x numel(P)) Jacobian dF/dP. Not computed when
%            nargout < 2.
%     info : struct
%              indices : nTar x 1 correspondences used
%              V       : nVerts x 3 deformed vertices
%              rows    : struct with 1x2 [first last] row ranges of F (and J)
%                        for fields fit, rot1, rot2, smooth, lap
%
%   Energy terms (scaling as in the original scripts):
%     Efit    = (V(indices,:) - tarV) / sqrt(nTar)                  nTar x 3
%     Erot1   = (A_i' A_i - I) / sqrt(n)                            3 x 3 x n
%     Erot2   = (det(A_i) - 1) / sqrt(n)                            1 x 1 x n
%     Esmooth = sum_j (A_i (g_j - g_i) + g_i + t_i - g_j - t_j) / sqrt(3n)   3 x n
%     Elap    = (Lap * V) ./ diag(Lap)                              nVerts x 3
%
%   The Jacobian is the corrected one unless cfg.legacyJacobian is true, which
%   reproduces the Jacobian of the original scripts including its errors (F is
%   identical in both modes).
%
%   Parameter layout inside P: r = 1:3, T = 4:6, A = 7:(6+9n), t = (7+9n):end.

cfg = model.cfg;
legacy = isfield(cfg, 'legacyJacobian') && cfg.legacyJacobian;   % absent -> corrected
w_rot = cfg.wRot;
w_smooth = cfg.wSmooth;
w_lap = cfg.wLap;
Lap = model.Lap;
idxNodeNeighbor = model.idxNodeNeighbor;   % Kn x n
n_nodes = model.nNodes;
n_verts = model.nVerts;

[R0, T0, A0, t0, J_R0] = edsm.decodeParam(P, n_nodes);   % J_R0 : 3 x 9
g0 = model.g0;                                           % 3 x n

d = deformCore(R0, T0, A0, t0, model);
deformedV = d.V;                                         % nVerts x 3

if nargin < 3 || isempty(indices)
    indices = edsm.nearestIdx(deformedV, model.tar.V);   % nTar x 1
end
indices = indices(:);
n_target = size(model.tar.V, 1);

% ---- residual terms ------------------------------------------------------
Efit = (deformedV(indices, :) - model.tar.V) ./ sqrt(n_target);               % nTar x 3

Elap = (Lap * deformedV) ./ model.lapDiag;                                    % nVerts x 3

Erot1 = (pagemtimes(A0, 'transpose', A0, 'none') - reshape(eye(3), 3, 3, 1)) ./ sqrt(n_nodes);   % 3 x 3 x n
Erot2 = (pagedet(A0) - 1.0) ./ sqrt(n_nodes);                                                    % 1 x 1 x n

g_node1 = reshape(g0(:, idxNodeNeighbor'), [size(g0, 1) size(idxNodeNeighbor')]);   % 3 x n x Kn
t_node1 = reshape(t0(:, idxNodeNeighbor'), [size(t0, 1) size(idxNodeNeighbor')]);   % 3 x n x Kn
v_node1 = reshape(g_node1 - g0, [size(g_node1, 1) 1 size(g_node1, 2:3)]);           % 3 x 1 x n x Kn
Esmooth = sum(squeeze(pagemtimes(A0, v_node1)) + g0 + t0 - g_node1 - t_node1, 3) ./ sqrt(3 * n_nodes);   % 3 x n

F = [Efit(:); w_rot*Erot1(:); w_rot*Erot2(:); w_smooth*Esmooth(:); w_lap* Elap(:)];

if nargout > 2
    info.indices = indices;
    info.V = deformedV;
    r1 = numel(Efit);
    r2 = r1 + numel(Erot1);
    r3 = r2 + numel(Erot2);
    r4 = r3 + numel(Esmooth);
    r5 = r4 + numel(Elap);
    info.rows = struct('fit', [1 r1], 'rot1', [r1 + 1, r2], 'rot2', [r2 + 1, r3], ...
        'smooth', [r3 + 1, r4], 'lap', [r4 + 1, r5]);
end

if nargout < 2
    return
end

% ---- Jacobian ------------------------------------------------------------
nP = numel(P);
nA = 9 * n_nodes;
lap = Lap;

% Derivative of the deformed vertices (3*nVerts rows: x block, y block, z block)
JdeformedV_R = jacDeformedR(d, J_R0, n_verts);               % 3nV x 3
JdeformedV_A = jacDeformedA(d, R0, n_verts, nA);             % 3nV x 9n
JdeformedV_t = jacDeformedT(d, R0, n_verts, n_nodes);        % 3nV x 3n
JdeformedV_At = [JdeformedV_A, JdeformedV_t];                % 3nV x 12n

% valid(i) = false for vertices reset to the source position in F: their position
% does not depend on any parameter. The legacy Jacobian treats them as deformed.
valid = true(n_verts, 1);
if ~legacy
    valid = ~d.invalid(:);
    keep = spdiags(repmat(double(valid), [3 1]), 0, 3 * n_verts, 3 * n_verts);
    JdeformedV_R = keep * JdeformedV_R;
    JdeformedV_At = keep * JdeformedV_At;
end

Jfit = jacFit(JdeformedV_R, JdeformedV_At, valid, indices, n_verts, n_target, numel(Efit));
Jlap = jacLap(JdeformedV_R, JdeformedV_At, valid, lap, n_verts, numel(Elap), legacy);
Jsmooth = jacSmooth(v_node1, idxNodeNeighbor, n_nodes, nA, numel(Esmooth));
Jrot1 = jacRot1(A0, n_nodes, numel(Erot1), nP, legacy);
Jrot2 = jacRot2(A0, n_nodes, numel(Erot2), nP, legacy);

J = [Jfit ./ sqrt(n_target);
    w_rot * Jrot1 ./ sqrt(n_nodes);
    w_rot * Jrot2 ./ sqrt(n_nodes);
    w_smooth * Jsmooth ./ sqrt(3 * n_nodes);
    w_lap * Jlap ./ reshape(repmat(model.lapDiag, [3 1]), [], 1)];
end

% =========================================================================
%  Jacobian of the deformed vertices
%  V = (local) * R' + T + G0,  local_i = sum_k w_ik (A_k (v_i - g_k) + g_k + t_k) - G0
%  Rows of the outputs: (coordinate j, vertex i) -> (j-1)*nVerts + i.
% =========================================================================
function JdeformedV_R = jacDeformedR(d, J_R0, n_verts)
% dV_ij / dR_jk = local_ik, chained with dR/dr (J_R0, converted to column-major).
J_R0 = sparse(J_R0(:, [1 4 7 2 5 8 3 6 9]));     % 3 x 9, columns = R(:) order

deformedV_R_col = reshape(repmat([1 4 7 2 5 8 3 6 9], [n_verts 1]), [], 1);              % 9nV x 1, column of R(:) element
deformedV_R_row = reshape(repmat(reshape(1:3*n_verts, [], 3), [3, 1]), [], 1);           % 9nV x 1, row in 3nV
deformedV_R_val = repmat(reshape(d.local, [], 1), [3 1]);                                % 9nV x 1, local_ik

JdeformedV_R = sparse(deformedV_R_row, deformedV_R_col, deformedV_R_val, 3 * n_verts, 9) * J_R0';   % 3nV x 3
end

function JdeformedV_A = jacDeformedA(d, R0, n_verts, nA)
% dV_ij / dA_k(c,m) = R_jc * w_ik * (v_i - g_k)_m, for the K nodes k of vertex i.
idxNeighbor = d.idxNeighbor;                     % K x nV
distNeighbor_ = d.distNeighbor_;                 % 1 x nV x K
v1 = d.v1;                                       % 3 x 1 x nV x K
n_neighbor = size(idxNeighbor, 1);

deformedV_A_col = reshape(repmat(reshape(repmat(9 * (reshape(idxNeighbor, 1, []) - 1), [9 1]) + [1; 2; 3; 4; 5; 6; 7; 8; 9], 1, []), [3 1]), [], 1);   % 27 K nV x 1
deformedV_A_row = reshape(repmat(reshape(1:3*n_verts, [], 3)', [3 * 3 * n_neighbor, 1]), [], 1);                                                         % 27 K nV x 1
deformedV_A_val = repelem(reshape(permute(distNeighbor_ .* squeeze(v1), [1 3 2]), 3, []), 9, 1);   % 27 x (K nV)
deformedV_A_val = repmat(R0(:), [3 1]) .* deformedV_A_val;                                         % 27 x (K nV)

JdeformedV_A = sparse(deformedV_A_row, deformedV_A_col, deformedV_A_val(:), 3 * n_verts, nA);
end

function JdeformedV_t = jacDeformedT(d, R0, n_verts, n_nodes)
% dV_ij / dt_k(c) = R_jc * w_ik
idxNeighbor = d.idxNeighbor;                     % K x nV
distNeighbor_ = d.distNeighbor_;                 % 1 x nV x K
n_neighbor = size(idxNeighbor, 1);

deformedV_t_col = repelem(reshape(repmat(3 * (reshape(idxNeighbor, 1, []) - 1), [3 1]) + [1; 2; 3], [], 1), 3, 1);   % 9 K nV x 1
deformedV_t_row = reshape(repmat(reshape(1:3*n_verts, [], 3)', [3 * n_neighbor, 1]), [], 1);                         % 9 K nV x 1
deformedV_t_val = reshape(R0(:) .* repelem(reshape(squeeze(permute(distNeighbor_, [1 3 2])), 1, []), 9, 1), [], 1);   % 9 K nV x 1

JdeformedV_t = sparse(deformedV_t_row, deformedV_t_col, deformedV_t_val, 3 * n_verts, 3 * n_nodes);
end

% =========================================================================
%  Per-term Jacobians (unscaled; scaling is applied when stacking in J)
% =========================================================================
function Jfit = jacFit(JdeformedV_R, JdeformedV_At, valid, indices, n_verts, n_target, nFit)
% Rows of the fit term pick the deformed-vertex rows of the matched vertices.
% d/dT is the identity per coordinate block, except for matched invalid vertices
% (all ones when valid is all true, i.e. in legacy mode).
m = double(valid(indices));
Jfit_T = spalloc(nFit, 3, nFit);                 % 3nTar x 3
Jfit_T(1:n_target, 1) = m;
Jfit_T((n_target + 1):(2 * n_target), 2) = m;
Jfit_T((2 * n_target + 1):(3 * n_target), 3) = m;

rowsSel = [indices n_verts+indices 2*n_verts+indices];     % nTar x 3 row selector, column-major flattened
Jfit = [JdeformedV_R(rowsSel, :), Jfit_T, JdeformedV_At(rowsSel, :)];   % 3nTar x nP
end

function Jlap = jacLap(JdeformedV_R, JdeformedV_At, valid, lap, n_verts, nLap, legacy)
% Elap = Lap * V per coordinate block.
Jlap_At = [lap*JdeformedV_At(1:n_verts, :);
    lap*JdeformedV_At(n_verts+(1:n_verts), :);
    lap*JdeformedV_At(2*n_verts+(1:n_verts), :)];             % 3nV x 12n
if legacy
    % Legacy: r and T columns left zero (misses dV/dr; the T part is only zero
    % while every vertex is deformed, since Lap has zero row sums).
    Jlap = [sparse(nLap, 6), Jlap_At];                        % 3nV x nP
else
    Jlap_R = [lap*JdeformedV_R(1:n_verts, :);
        lap*JdeformedV_R(n_verts+(1:n_verts), :);
        lap*JdeformedV_R(2*n_verts+(1:n_verts), :)];          % 3nV x 3
    % dV_i/dT = e_c for valid vertices only: nonzero next to invalid vertices.
    Jlap_T = [lap*double(valid), sparse(n_verts, 2); ...
        sparse(n_verts, 1), lap*double(valid), sparse(n_verts, 1); ...
        sparse(n_verts, 2), lap*double(valid)];               % 3nV x 3
    Jlap = [Jlap_R, Jlap_T, Jlap_At];                         % 3nV x nP
end
end

function Jsmooth = jacSmooth(v_node1, idxNodeNeighbor, n_nodes, nA, nSmooth)
% Esmooth row (c, i): sum_j A_i(c,:) (g_j - g_i) + ... ; rows (i-1)*3 + c.
% d/dA_i(c,m) = sum_j (g_j - g_i)_m ; d/dt_i = +Kn ; d/dt_j = -1 for each listed j (duplicates accumulate).
% (The repmat factor [3, 1] below is the number of columns of A_i.)
smooth_A_row = reshape(repmat(reshape(1:nSmooth, 3, []), [3, 1]), [], 1);                         % 9n x 1
smooth_A_col = reshape((1:nA), [], 1);                                                            % 9n x 1
smooth_A_val = reshape(repmat(reshape(squeeze(sum(v_node1, 4)), 1, []), [3, 1]), [], 1);          % 9n x 1

Jsmooth_A = sparse(smooth_A_row, smooth_A_col, smooth_A_val, nSmooth, nA);

Kn = size(idxNodeNeighbor, 1);
smooth_t_row = reshape(repmat(1:3*n_nodes, [Kn, 1]), [], 1);                                        % 3 n Kn x 1
smooth_t_col = repmat(3 * idxNodeNeighbor, [3 1]) + repmat(repelem([-2; -1; 0], Kn), [1 n_nodes]);  % 3Kn x n
smooth_t_val = repmat(-1, [3 * n_nodes * Kn, 1]);                                                   % 3 n Kn x 1

% +Kn on the node's own t (independent of where, or whether, the node is listed
% among its neighbors); sparse() accumulates duplicates.
Jsmooth_t = sparse([smooth_t_row; (1:3*n_nodes)'], [smooth_t_col(:); (1:3*n_nodes)'], ...
    [smooth_t_val; repmat(Kn, [3 * n_nodes, 1])], nSmooth, 3 * n_nodes);

Jsmooth = [sparse(nSmooth, 6), Jsmooth_A, Jsmooth_t];       % 3n x nP
end

function Jrot1 = jacRot1(A0, n_nodes, nRot1, nP, legacy)
% Erot1 = A'A - I, 9 entries per node, rows (node i) -> 9(i-1) + [1..9];
% each row depends on 5 entries of the node's A (45 nonzeros per node).
% The values are derivatives w.r.t. the row-major A_i(:); P stores A_i(:) column-major,
% so they are placed in the column-major slots (legacy: left in row-major slots).
rot1_row =[1 2 3 4 7; 2 4 5 6 8; 3 6 7 8 9; 1 2 3 4 7; 2 4 5 6 8; 3 6 7 8 9; 1 2 3 4 7; 2 4 5 6 8; 3 6 7 8 9]';   % 5 x 9
rot1_rows = repmat(reshape(rot1_row, [], 1), [1 n_nodes])  + 9 * ((1:n_nodes) - 1);                              % 45 x n

slots = 1:9;                                                 % legacy: row-major order, wrong for non-symmetric A_i
if ~legacy, slots = [1 4 7 2 5 8 3 6 9]; end                 % row-major element k -> column-major slot
rot1_cols = reshape(reshape(repmat(slots + 6, [5 n_nodes]), [], n_nodes) + repmat(9 * ((1:n_nodes) - 1), [45 1]), [], 1);   % 45 n x 1

A = reshape(permute(A0, [2 1 3]), [], size(A0, 3));          % 9 x n, row-major A_i(:)
rot1_vals = [2 * A(1, :); A(2, :); A(3, :); A(2, :); A(3, :);
    A(1, :); A(1, :); 2 * A(2, :); A(3, :); A(3, :);
    A(1, :); A(2, :); A(1, :); A(2, :); 2 * A(3, :);
    2 * A(4, :); A(5, :); A(6, :); A(5, :); A(6, :);
    A(4, :); A(4, :); 2 * A(5, :); A(6, :); A(6, :);
    A(4, :); A(5, :); A(4, :); A(5, :); 2 * A(6, :);
    2 * A(7, :); A(8, :); A(9, :); A(8, :); A(9, :);
    A(7, :); A(7, :); 2 * A(8, :); A(9, :); A(9, :);
    A(7, :); A(8, :); A(7, :); A(8, :); 2 * A(9, :)];        % 45 x n
rot1_vals = reshape(rot1_vals, [], 1);

Jrot1 = sparse(rot1_rows, rot1_cols, rot1_vals, nRot1, nP);   % 9n x nP
end

function Jrot2 = jacRot2(A0, n_nodes, nRot2, nP, legacy)
% Erot2 = det(A_i) - 1, one row per node, derivative = cofactors of A_i.
% Same row-major/column-major placement as jacRot1 (legacy: row-major slots).
rot2_rows = reshape(repmat(1:n_nodes, [9 1]), [], 1);                                                            % 9n x 1

slots = 1:9;
if ~legacy, slots = [1 4 7 2 5 8 3 6 9]; end
rot2_cols = reshape(reshape(repmat(slots + 6, [1 n_nodes]), [], n_nodes) + repmat(9 * ((1:n_nodes) - 1), [9 1]), [], 1);   % 9n x 1

A = reshape(permute(A0, [2 1 3]), [], size(A0, 3));          % 9 x n, row-major A_i(:)
rot2_vals = [A(5, :) .* A(9, :) - A(6, :) .* A(8, :);
    A(6, :) .* A(7, :) - A(4, :) .* A(9, :);
    A(4, :) .* A(8, :) - A(5, :) .* A(7, :);
    A(3, :) .* A(8, :) - A(2, :) .* A(9, :);
    A(1, :) .* A(9, :) - A(3, :) .* A(7, :);
    A(2, :) .* A(7, :) - A(1, :) .* A(8, :);
    A(2, :) .* A(6, :) - A(3, :) .* A(5, :);
    A(3, :) .* A(4, :) - A(1, :) .* A(6, :);
    A(1, :) .* A(5, :) - A(2, :) .* A(4, :)];                % 9 x n
rot2_vals = reshape(rot2_vals, [], 1);

Jrot2 = sparse(rot2_rows, rot2_cols, rot2_vals, nRot2, nP);   % n x nP
end

function D = pagedet(X)
% Determinant of every 3x3 page of X (3 x 3 x n -> 1 x 1 x n).
D = X(1, 1, :).*X(2, 2, :).*X(3, 3, :) + X(1, 2, :).*X(2, 3, :).*X(3, 1, :) + X(1, 3, :).*X(2, 1, :).*X(3, 2, :) ...
    - X(1, 3, :).*X(2, 2, :).*X(3, 1, :) - X(1, 2, :).*X(2, 1, :).*X(3, 3, :) - X(1, 1, :).*X(2, 3, :).*X(3, 2, :);
end
