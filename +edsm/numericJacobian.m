function J = numericJacobian(P, model, opts)
%NUMERICJACOBIAN  Central finite-difference Jacobian of edsm.residual (for verification).
%
%   J = edsm.numericJacobian(P, model)
%   J = edsm.numericJacobian(P, model, opts)
%
%   Inputs:
%     P     : (6 + 12n) parameter vector
%     model : struct from edsm.buildModel
%     opts  : (optional) struct with fields
%               step     : relative step, default 1e-6; the step of column k is
%                          step * max(1, abs(P(k)))
%               cols     : indices of the columns to compute, default 1:numel(P)
%               parallel : true to use parfor (default false)
%   Output:
%     J : numel(F) x numel(cols) FULL matrix, J(:,c) = dF/dP(cols(c)).
%
%   The nearest-neighbor correspondences are computed once at P and held fixed
%   while P is perturbed, matching the analytic Jacobian (which treats them as
%   constants). Memory: numel(F) x numel(cols) doubles (about 150 MB for the
%   full Jacobian of a 5027-vertex mesh with 50 nodes).

if nargin < 3, opts = struct(); end
step = getOpt(opts, 'step', 1e-6);
cols = getOpt(opts, 'cols', 1:numel(P));
parallel = getOpt(opts, 'parallel', false);

P = P(:);
[F0, ~, info] = edsm.residual(P, model);
indices = info.indices;

J = zeros(numel(F0), numel(cols));
if parallel
    parfor c = 1:numel(cols)
        J(:, c) = column(P, model, indices, cols(c), step);
    end
else
    for c = 1:numel(cols)
        J(:, c) = column(P, model, indices, cols(c), step);
    end
end
end

function col = column(P, model, indices, k, step)
h = step * max(1, abs(P(k)));
Pp = P; Pp(k) = Pp(k) + h;
Pm = P; Pm(k) = Pm(k) - h;
Fp = edsm.residual(Pp, model, indices);
Fm = edsm.residual(Pm, model, indices);
col = (Fp - Fm) / (2 * h);
end

function v = getOpt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name))
    v = opts.(name);
else
    v = default;
end
end
