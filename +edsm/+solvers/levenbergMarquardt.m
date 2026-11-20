function [P, hist] = levenbergMarquardt(P0, model, opts)
%LEVENBERGMARQUARDT  Levenberg-Marquardt with accept/reject lambda control (legacy LM_hjh_2, CG, PCG).
%
%   [P, hist] = edsm.solvers.levenbergMarquardt(P0, model)
%   [P, hist] = edsm.solvers.levenbergMarquardt(P0, model, opts)
%
%   Inputs:
%     P0    : (6 + 12n) start parameters
%     model : struct from edsm.buildModel
%     opts  : struct, any field may be omitted
%               maxIter       100      maximum number of ACCEPTED iterations
%               lambda0       0.01     initial damping
%               lambdaMax     1e5      stop when lambda exceeds this after a rejection
%               damping       'diag'   'diag' : A = J'J + lambda*diag(diag(J'J))
%                                      'eye'  : A = J'J + lambda*I
%                                      'none' : A = J'J (lambda has no effect)
%               linearSolver  'direct' 'direct' (A\b) | 'cg' | 'pcg' (Jacobi) via edsm.solvers.conjGrad
%               cgTol         1e-10    conjGrad tol (absolute residual norm)
%               cgMaxIter     numel(P) conjGrad maxIter
%               gradTol       1e-6     stop when norm(2*J'*F, inf) <= gradTol * (same norm at P0)
%                                      (relative to the INITIAL gradient); 0 disables
%               funTol        1e-8     stop after 3 consecutive accepted steps that each reduce
%                                      err by less than funTol*(1 + err_new); 0 disables
%               verbose       false    print one line per iteration
%               callback      []       @(iter, P, hist), called after each accepted iteration
%   Outputs:
%     P    : parameters after the last accepted step
%     hist : struct
%              err        (nIter+1)x1  sum(F.^2); err(1) at P0, err(k+1) after iteration k
%              gradNorm   (nIter+1)x1  norm(2*J'*F, inf), the gradient of sum(F.^2), at P0 and
%                                      after each accepted iteration (same indexing as err)
%              lambda     nIterx1      damping that produced the accepted step k
%                                      (NaN for damping 'none')
%              lambdaNext nIterx1      damping carried into iteration k+1
%              nReject    nIterx1      rejected trial steps before step k was accepted
%              stepNorm   nIterx1      2-norm of the accepted step
%              linIters   nSolvex1     CG iterations of every linear solve (all
%                                      trials, in order; empty for 'direct')
%              time       (nIter+1)x1  cumulative seconds
%              nIter      accepted iterations
%              stopReason 'gradTol' | 'funTol' | 'maxIter' | 'lambdaMax' | 'rejectedNoDamping'
%
%   One iteration (g = J'*F, evaluated at the current accepted point):
%     repeat
%       build A with the CURRENT lambda; solve A*delta = -g
%       trial = P + delta; accept if sum(F(trial).^2) < err (strict decrease)
%       on reject: lambda <- 10*lambda; stop if lambda > lambdaMax
%     until accepted
%     lambda <- lambda/10 only if the first trial was accepted (no rejection).
%   A non-finite delta counts as a rejection. The matrix is rebuilt with the
%   new lambda on every retry (the legacy CG/PCG scripts build it once before
%   the retry loop; see the regression test).
%
%   Stopping (checked after every accepted iteration, in this order; gradTol is
%   also checked at P0 before the first iteration, giving nIter = 0):
%     'gradTol'    gradNorm(end) <= gradTol * gradNorm(1). The test is relative to
%                  the initial gradient, so it does not depend on the scale of
%                  sum(F.^2); if P0 is already stationary (gradNorm(1) = 0) it
%                  stops at once. With the legacy Jacobian the gradient does not
%                  go to zero (it is not the gradient of err), so this never fires.
%     'funTol'     the last 3 accepted steps each had
%                  err_old - err_new < funTol*(1 + err_new). One small step is not
%                  enough: recorded histories contain isolated tiny steps followed
%                  by renewed progress. Rejected trial steps do not count.
%     'maxIter'    maxIter accepted steps (also the reason when a criterion above
%                  would only be met after the last allowed iteration)
%     'lambdaMax'  lambda > lambdaMax after a rejection, P stays at the last
%                  accepted point
%     'rejectedNoDamping'  a step is rejected with damping 'none', where retrying
%                  cannot change the system.
%   gradTol = 0 / funTol = 0 switch the respective test off (the pre-criteria
%   behavior). The Jacobian is evaluated at every accepted point, including the
%   last, because the gradient is needed there.

d = struct('maxIter', 100, 'lambda0', 0.01, 'lambdaMax', 1e5, ...
           'damping', 'diag', 'linearSolver', 'direct', ...
           'cgTol', 1e-10, 'cgMaxIter', numel(P0), ...
           'gradTol', 1e-6, 'funTol', 1e-8, ...
           'verbose', false, 'callback', []);
if nargin < 3, opts = struct(); end
o = parseOpts(opts, d, 'levenbergMarquardt');
if ~ismember(o.damping, {'diag', 'eye', 'none'})
    error('edsm:solvers:badOption', 'damping must be ''diag'', ''eye'' or ''none''.');
end
if ~ismember(o.linearSolver, {'direct', 'cg', 'pcg'})
    error('edsm:solvers:badOption', 'linearSolver must be ''direct'', ''cg'' or ''pcg''.');
end
cgOpts = struct('tol', o.cgTol, 'maxIter', o.cgMaxIter, ...
                'precond', struct('cg', 'none', 'pcg', 'jacobi', 'direct', 'none').(o.linearSolver));

t0 = tic;
P = P0(:);
lambda = o.lambda0;
nSmallNeeded = 3;               % consecutive small steps required by funTol
nSmall = 0;
[F, J] = evalResidual(P, model, 'analytic', true);
err = sum(F.^2);
hist = histInit(err, toc(t0));
hist.gradNorm = norm(2 * (J' * F), inf);
gradStop = o.gradTol * hist.gradNorm;
hist.linIters = zeros(0, 1);
hist.stopReason = 'maxIter';
if o.gradTol > 0 && hist.gradNorm <= gradStop
    hist.stopReason = 'gradTol';
    return
end

for k = 1:o.maxIter
    JtJ = J' * J;
    g = J' * F;
    n = numel(P);
    nReject = 0;
    accepted = false;
    while true
        A = dampedMatrix(JtJ, lambda, o.damping);
        if strcmp(o.linearSolver, 'direct')
            delta = A \ (-g);
        else
            [delta, res] = edsm.solvers.conjGrad(A, -g, zeros(n, 1), cgOpts);
            hist.linIters(end+1, 1) = numel(res);
        end
        Pt = P + delta;
        if all(isfinite(delta))
            Ft = evalResidual(Pt, model, 'analytic', false);
            errT = sum(Ft.^2);
        else
            errT = Inf;
        end
        if errT < err
            accepted = true;
            break
        end
        nReject = nReject + 1;
        if strcmp(o.damping, 'none')
            hist.stopReason = 'rejectedNoDamping';
            break
        end
        lambda = lambda * 10;
        if lambda > o.lambdaMax
            hist.stopReason = 'lambdaMax';
            break
        end
    end
    if ~accepted, break, end

    lambdaUsed = lambda;
    if nReject == 0, lambda = lambda / 10; end
    if strcmp(o.damping, 'none'), lambdaUsed = NaN; end
    P = Pt;
    errOld = err;
    err = errT;
    [F, J] = evalResidual(P, model, 'analytic', true);
    gradNorm = norm(2 * (J' * F), inf);
    hist = histPush(hist, err, norm(delta), toc(t0), ...
        struct('lambda', lambdaUsed, 'lambdaNext', lambda, 'nReject', nReject, 'gradNorm', gradNorm));
    reportIter('levenbergMarquardt', o, P, hist);

    if o.gradTol > 0 && gradNorm <= gradStop
        hist.stopReason = 'gradTol';
        break
    end
    if o.funTol > 0 && errOld - err < o.funTol * (1 + err)
        nSmall = nSmall + 1;
        if nSmall >= nSmallNeeded
            hist.stopReason = 'funTol';
            break
        end
    else
        nSmall = 0;
    end
end
end

function A = dampedMatrix(JtJ, lambda, damping)
n = size(JtJ, 1);
switch damping
    case 'diag'
        A = JtJ + lambda * spdiags(diag(JtJ), 0, n, n);
    case 'eye'
        A = JtJ + lambda * speye(n);
    case 'none'
        A = JtJ;
end
end
