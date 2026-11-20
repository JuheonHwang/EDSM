function h = histInit(err0, t0)
%HISTINIT  Start a solver history struct (shared solver helper).
%
%   h = histInit(err0, t0)
%
%   h.err(1)  = err0 = sum(F.^2) at the starting point P0
%   h.time(1) = t0   = seconds spent until then (initial evaluation)
%   h.stepNorm is empty; h.nIter = 0; h.stopReason = ''.
%   Per-iteration vectors are appended with histPush.

h = struct('err', err0, 'stepNorm', zeros(0, 1), 'time', t0, ...
           'nIter', 0, 'stopReason', '');
end
