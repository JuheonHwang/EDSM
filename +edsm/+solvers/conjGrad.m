function [x, resHist] = conjGrad(A, b, x0, opts)
%CONJGRAD  (Preconditioned) conjugate gradients for A*x = b, A symmetric positive definite.
%
%   [x, resHist] = edsm.solvers.conjGrad(A, b, x0)
%   [x, resHist] = edsm.solvers.conjGrad(A, b, x0, opts)
%
%   Inputs:
%     A    : n x n SPD matrix (sparse or full), or anything that supports A*v
%            (Jacobi preconditioning needs diag(A), so a matrix there)
%     b    : n x 1 right-hand side
%     x0   : start vector ([] = zeros)
%     opts : struct, any field may be omitted
%              precond  'none'   'none' | 'jacobi' (M = diag(A); entries <= 0 are
%                                replaced by 1)
%              tol      1e-10    stop when norm(r) < tol (ABSOLUTE 2-norm of the
%                                true residual r = b - A*x, as in the legacy code)
%              maxIter  numel(b) maximum number of CG iterations
%   Outputs:
%     x       : approximate solution
%     resHist : k x 1, norm(r) after each of the k performed iterations
%               (k = numel(resHist) is the iteration count; empty if x0 is
%               already within tol)
%
%   Stopping: norm(r) < tol, maxIter reached, or p'*A*p <= 0 (A is not SPD;
%   the iteration is abandoned before the update).
%   Reproduces the legacy inline CG (precond 'none') and preconjgrad
%   ('jacobi') operation for operation.

d = struct('precond', 'none', 'tol', 1e-10, 'maxIter', numel(b));
if nargin < 4, opts = struct(); end
o = parseOpts(opts, d, 'conjGrad');

b = b(:);
if nargin < 3 || isempty(x0), x0 = zeros(size(b)); end
x = x0(:);

switch o.precond
    case 'none'
        dg = [];
    case 'jacobi'
        dg = full(diag(A));
        dg(dg <= 0) = 1;
    otherwise
        error('edsm:solvers:badOption', 'conjGrad: precond must be ''none'' or ''jacobi'' (got ''%s'').', string(o.precond));
end

r = b - A * x;
resHist = zeros(0, 1);
if norm(r) < o.tol, return, end

z = precondition(r, dg);
p = z;
rz = r' * z;
for i = 1:o.maxIter
    Ap = A * p;
    pAp = p' * Ap;
    if pAp <= 0, break, end
    alpha = rz / pAp;
    x = x + alpha * p;
    r = r - alpha * Ap;
    rn = sqrt(r' * r);
    resHist(end+1, 1) = rn; %#ok<AGROW>
    if rn < o.tol, break, end
    z = precondition(r, dg);
    rzNew = r' * z;
    p = z + (rzNew / rz) * p;
    rz = rzNew;
end
end

function z = precondition(r, dg)
if isempty(dg), z = r; else, z = r ./ dg; end
end
