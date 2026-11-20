function b = meshBounds(varargin)
%MESHBOUNDS  2x3 [min; max] of the union of several n x 3 vertex arrays.
mn = inf(1, 3); mx = -inf(1, 3);
for k = 1:nargin
    mn = min(mn, min(varargin{k}, [], 1));
    mx = max(mx, max(varargin{k}, [], 1));
end
b = [mn; mx];
end
