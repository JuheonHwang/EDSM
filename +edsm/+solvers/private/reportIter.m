function reportIter(name, o, P, h)
%REPORTITER  Per-iteration print and callback (shared solver helper).
%
%   reportIter(name, o, P, h)
%
%   Called by every solver after each completed iteration. If o.verbose a
%   one-line summary of the last history entry is printed; if o.callback is
%   non-empty it is called as o.callback(h.nIter, P, h). The solvers never
%   plot; visualization is the callback's job.

k = h.nIter;
if o.verbose
    s = sprintf('[%s] iter %3d  err %.10g  |step| %.3e  t %.2fs', ...
        name, k, h.err(end), h.stepNorm(end), h.time(end));
    if isfield(h, 'lambda'), s = sprintf('%s  lambda %.3g', s, h.lambda(end)); end
    fprintf('%s\n', s);
end
if ~isempty(o.callback)
    o.callback(k, P, h);
end
end
