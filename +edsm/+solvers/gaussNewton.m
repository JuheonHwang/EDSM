function [P, hist] = gaussNewton(P0, model, opts)
%GAUSSNEWTON  Undamped Gauss-Newton with minimum-norm step (legacy Gauss_Newton_hjh_re).
%
%   [P, hist] = edsm.solvers.gaussNewton(P0, model)
%   [P, hist] = edsm.solvers.gaussNewton(P0, model, opts)
%
%   Inputs:
%     P0    : (6 + 12n) start parameters
%     model : struct from edsm.buildModel
%     opts  : struct, any field may be omitted
%               maxIter     100    maximum number of iterations
%               stepTol     1e-3   stop when norm(delta, inf) < stepTol
%               jacobian    'analytic'  'analytic' (edsm.residual) | 'numeric'
%                                       (edsm.numericJacobian, central differences)
%               numericOpts struct()    options passed to edsm.numericJacobian
%               rankTol     []     absolute tolerance passed to lsqminnorm (default: its own,
%                                  eps*max(size(J))*norm(J)). A finite-difference J has noise
%                                  of order 1e-8 in the exactly-null singular values (the
%                                  analytic J has 1e-15), which the default does not truncate:
%                                  use rankTol of about 1e-5 with jacobian = 'numeric'.
%               verbose     false  print one line per iteration
%               callback    []     @(iter, P, hist), called after each iteration
%   Outputs:
%     P    : final parameters
%     hist : struct
%              err        (nIter+1)x1  sum(F.^2); err(1) at P0, err(k+1) after iteration k
%              stepNorm   nIterx1      2-norm of delta in iteration k
%              time       (nIter+1)x1  cumulative seconds
%              nIter      iterations performed
%              stopReason 'stepTol' | 'maxIter' | 'nonFinite'
%
%   Iteration: delta = lsqminnorm(J, -F); P <- P + delta. There is no damping
%   and no line search, so the method can diverge (the error is not
%   monotone).
%
%   Stopping criteria, checked after the step of iteration k is applied:
%     'stepTol'    norm(delta, inf) < stepTol
%     'maxIter'    k == maxIter
%     'nonFinite'  err is NaN or Inf
%   Legacy mapping: the legacy test
%       (|F-Fprev| < eps*(1+F)) | (norm(Grad,inf) < eps3*(1+F)) | (norm(delta,inf) < eps2*(1+norm(delta,inf)))
%   compares vectors elementwise in its first two terms (F is a vector), so
%   those fire only when ALL components satisfy them. The third term is
%   equivalent to norm(delta,inf) < eps2/(1-eps2) = 1.001001e-3, i.e.
%   stepTol = 1e-3/(1-1e-3). The default stepTol = 1e-3 differs from that by
%   0.1 percent; pass opts.stepTol = 1e-3/(1-1e-3) for the exact legacy value.

d = struct('maxIter', 100, 'stepTol', 1e-3, 'jacobian', 'analytic', ...
           'numericOpts', struct(), 'rankTol', [], 'verbose', false, 'callback', []);
if nargin < 3, opts = struct(); end
o = parseOpts(opts, d, 'gaussNewton');

t0 = tic;
P = P0(:);
[F, J] = evalResidual(P, model, o.jacobian, o.maxIter > 0, o.numericOpts);
hist = histInit(sum(F.^2), toc(t0));
hist.stopReason = 'maxIter';

for k = 1:o.maxIter
    if isempty(o.rankTol), delta = lsqminnorm(J, -F); else, delta = lsqminnorm(J, -F, o.rankTol); end
    P = P + delta;
    [F, J] = evalResidual(P, model, o.jacobian, k < o.maxIter, o.numericOpts);
    err = sum(F.^2);
    hist = histPush(hist, err, norm(delta), toc(t0));
    reportIter('gaussNewton', o, P, hist);
    if ~isfinite(err)
        hist.stopReason = 'nonFinite';
        break
    end
    if norm(delta, inf) < o.stepTol
        hist.stopReason = 'stepTol';
        break
    end
end
end
