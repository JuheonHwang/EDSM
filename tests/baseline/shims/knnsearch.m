function [idx, d] = knnsearch(X, Y, varargin)
% Shim for Statistics Toolbox knnsearch, supports only 'K',1 (Euclidean).
% For each row of Y returns index of nearest row of X. Ties -> lowest index.
K = 1;
for k = 1:2:numel(varargin)
    if strcmpi(varargin{k}, 'K'), K = varargin{k+1}; end
end
assert(K == 1, 'knnsearch shim supports only K = 1');
X = double(gather(X)); Y = double(gather(Y));
nY = size(Y, 1);
idx = zeros(nY, 1); d = zeros(nY, 1);
chunk = 2000;
xx = sum(X.^2, 2)';
for s = 1:chunk:nY
    e = min(s + chunk - 1, nY);
    Yc = Y(s:e, :);
    D2 = sum(Yc.^2, 2) + xx - 2 * (Yc * X');
    [m, i] = min(D2, [], 2);
    idx(s:e) = i;
    d(s:e) = sqrt(max(m, 0));
end
end
