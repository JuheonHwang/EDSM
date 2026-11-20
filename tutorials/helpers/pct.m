function q = pct(x, p)
%PCT  Percentile of a vector (replaces prctile, no Statistics Toolbox needed).
%
%   q = pct(x, p)
%
%   x : vector; NaN entries are ignored.  p : percentage(s) in [0, 100].
%   Same definition as prctile: the k-th sorted value (k = 1..n) sits at
%   percentage 100*(k - 0.5)/n; values in between are interpolated linearly
%   and p outside [100*0.5/n, 100*(n - 0.5)/n] returns the min / max.
x = sort(x(~isnan(x(:))));
n = numel(x);
pos = min(max(p(:)' / 100 * n + 0.5, 1), n);
lo = floor(pos); hi = ceil(pos);
q = x(lo)' .* (1 - (pos - lo)) + x(hi)' .* (pos - lo);
q = reshape(q, size(p));
end
