function [P, hist] = gradientDescent(P0, model, opts)
%GRADIENTDESCENT  Fixed-step gradient descent on sum(F.^2) (legacy gradient_hjh).
%
%   [P, hist] = edsm.solvers.gradientDescent(P0, model)
%   [P, hist] = edsm.solvers.gradientDescent(P0, model, opts)
%
%   Inputs:
%     P0    : (6 + 12n) start parameters (e.g. edsm.initParam(model))
%     model : struct from edsm.buildModel
%     opts  : struct, any field may be omitted
%               maxIter         100    number of iterations
%               stepSize        1e-3   step on the 12n local parameters
%               globalStepScale 0.01   factor on stepSize for the 6 global
%                                      r/T parameters (P(1:6))
%               verbose         false  print one line per iteration
%               callback        []     @(iter, P, hist), called after each iteration
%   Outputs:
%     P    : final parameters
%     hist : struct
%              err        (nIter+1)x1  sum(F.^2); err(1) at P0, err(k+1) after iteration k
%              stepNorm   nIterx1      2-norm of the update applied in iteration k
%              time       (nIter+1)x1  cumulative seconds; time(1) = initial evaluation
%              nIter      number of iterations performed
%              stopReason 'maxIter' | 'nonFinite'
%
%   Update (g = 2*J'*F, the gradient of sum(F.^2); correspondences are
%   recomputed at every iterate):
%       P(1:6)     <- P(1:6)     - stepSize*globalStepScale * g(1:6)
%       P(7:end)   <- P(7:end)   - stepSize                 * g(7:end)
%
%   Stopping: maxIter reached, or the error becomes non-finite (the iterate
%   that produced it is kept). No convergence test (the legacy script has none).

d = struct('maxIter', 100, 'stepSize', 1e-3, 'globalStepScale', 0.01, ...
           'verbose', false, 'callback', []);
if nargin < 3, opts = struct(); end
o = parseOpts(opts, d, 'gradientDescent');

t0 = tic;
P = P0(:);
[F, J] = evalResidual(P, model, 'analytic', o.maxIter > 0);
hist = histInit(sum(F.^2), toc(t0));
hist.stopReason = 'maxIter';

scale = [o.globalStepScale * ones(6, 1); ones(numel(P) - 6, 1)];
for k = 1:o.maxIter
    g = 2 * (J' * F);
    step = -o.stepSize * scale .* g;
    P = P + step;
    [F, J] = evalResidual(P, model, 'analytic', k < o.maxIter);
    err = sum(F.^2);
    hist = histPush(hist, err, norm(step), toc(t0));
    reportIter('gradientDescent', o, P, hist);
    if ~isfinite(err)
        hist.stopReason = 'nonFinite';
        break
    end
end
end
