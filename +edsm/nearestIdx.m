function idx = nearestIdx(X, Y)
%NEARESTIDX  Nearest-neighbor search (replaces knnsearch(X, Y, 'K', 1)).
%
%   idx = edsm.nearestIdx(X, Y)
%
%   Inputs:
%     X : nx x d reference points
%     Y : ny x d query points
%   Output:
%     idx : ny x 1 column; idx(j) = row of X nearest (Euclidean) to Y(j,:).
%           Ties go to the smallest row index of X.
%
%   Exact brute force on squared differences (no ||x||^2 - 2x.y + ||y||^2
%   expansion, to avoid cancellation). Queries are processed in chunks so the
%   temporary distance matrix never exceeds about 2e7 elements.

nx = size(X, 1);
ny = size(Y, 1);
d = size(X, 2);
assert(size(Y, 2) == d, 'X and Y must have the same number of columns.');

chunk = max(1, floor(2e7 / max(nx, 1)));
idx = zeros(ny, 1);
Xt = X';                                  % d x nx
for first = 1:chunk:ny
    last = min(first + chunk - 1, ny);
    Yc = Y(first:last, :);                % c x d
    D2 = (Yc(:, 1) - Xt(1, :)).^2;        % c x nx
    for k = 2:d
        D2 = D2 + (Yc(:, k) - Xt(k, :)).^2;
    end
    [~, idx(first:last)] = min(D2, [], 2);
end
end
