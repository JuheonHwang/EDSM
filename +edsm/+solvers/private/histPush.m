function h = histPush(h, err, stepNorm, t, extra)
%HISTPUSH  Append one iteration to a solver history (shared solver helper).
%
%   h = histPush(h, err, stepNorm, t)
%   h = histPush(h, err, stepNorm, t, extra)
%
%   err      : sum(F.^2) after the iteration       -> appended to h.err
%   stepNorm : 2-norm of the applied update        -> appended to h.stepNorm
%   t        : cumulative seconds                  -> appended to h.time
%   extra    : optional struct of scalar per-iteration quantities (e.g.
%              lambda); each field is appended to the same-named column of h
%              (created on first use).
%   h.nIter is kept equal to numel(h.stepNorm).

h.err(end+1, 1) = err;
h.stepNorm(end+1, 1) = stepNorm;
h.time(end+1, 1) = t;
h.nIter = numel(h.stepNorm);
if nargin > 4
    names = fieldnames(extra);
    for i = 1:numel(names)
        nm = names{i};
        if ~isfield(h, nm), h.(nm) = zeros(0, 1); end
        h.(nm)(end+1, 1) = extra.(nm);
    end
end
end
