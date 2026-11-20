function [F, J] = evalResidual(P, model, jacobian, needJ, numericOpts)
%EVALRESIDUAL  Residual and (optionally) Jacobian at P (shared solver helper).
%
%   [F, J] = evalResidual(P, model, jacobian, needJ, numericOpts)
%
%   jacobian : 'analytic' -> J from edsm.residual (corrected or legacy,
%                            according to model.cfg.legacyJacobian)
%              'numeric'  -> J = edsm.numericJacobian(P, model, numericOpts)
%                            (full matrix, correspondences held at P)
%   needJ    : false -> only F is computed and J = [].
%
%   F is always edsm.residual(P, model), i.e. correspondences are recomputed
%   at P exactly as in the legacy scripts.

if nargin < 5, numericOpts = struct(); end
if ~needJ
    F = edsm.residual(P, model);
    J = [];
    return
end
switch jacobian
    case 'analytic'
        [F, J] = edsm.residual(P, model);
    case 'numeric'
        F = edsm.residual(P, model);
        J = edsm.numericJacobian(P, model, numericOpts);
    otherwise
        error('edsm:solvers:badOption', 'jacobian must be ''analytic'' or ''numeric'' (got ''%s'').', string(jacobian));
end
end
