function [idx, d2] = knnBrute(X, Q, K)
%KNNBRUTE  K nearest rows of X for every row of Q (brute force, chunked).
%
%   [idx, d2] = knnBrute(X, Q, K)
%     X : nx x 3 reference points
%     Q : nq x 3 query points
%     K : number of neighbors
%   Outputs (both K x nq, ascending distance, ties -> smaller index first):
%     idx : row indices into X
%     d2  : SQUARED Euclidean distances
%
%   Replaces the (internal, Computer Vision Toolbox) multiQueryKNNSearchImpl
%   used by the originals for distanceOnMesh = false; like that function it
%   returns squared distances.

nx = size(X, 1);
nq = size(Q, 1);
idx = zeros(K, nq);
d2 = zeros(K, nq);
chunk = max(1, floor(2e7 / max(nx, 1)));
Xt = X';
for first = 1:chunk:nq
    last = min(first + chunk - 1, nq);
    Qc = Q(first:last, :);
    D2 = (Qc(:, 1) - Xt(1, :)).^2;          % c x nx
    D2 = D2 + (Qc(:, 2) - Xt(2, :)).^2;
    D2 = D2 + (Qc(:, 3) - Xt(3, :)).^2;
    [dk, ik] = mink(D2, K, 2);               % c x K
    idx(:, first:last) = ik';
    d2(:, first:last) = dk';
end
end
