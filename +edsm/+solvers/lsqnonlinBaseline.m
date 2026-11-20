function [P, hist] = lsqnonlinBaseline(P0, model, opts)
%LSQNONLINBASELINE  MATLAB lsqnonlin with the options of the legacy Jall_FINAL script.
%
%   [P, hist] = edsm.solvers.lsqnonlinBaseline(P0, model)
%   [P, hist] = edsm.solvers.lsqnonlinBaseline(P0, model, opts)
%
%   Requires the Optimization Toolbox. The analytic Jacobian of edsm.residual
%   is supplied ('SpecifyObjectiveGradient', true); no bounds; the default
%   algorithm (trust-region-reflective) is used.
%
%   Inputs:
%     P0    : (6 + 12n) start parameters
%     model : struct from edsm.buildModel
%     opts  : struct, any field may be omitted
%               maxIter      30     'MaxIterations'
%               functionTol  1e-10  'FunctionTolerance'
%               optimOptions {}     extra name/value pairs for optimoptions
%                                   (applied last, may override the above)
%               verbose      false  Display 'iter' and one line per iteration
%               callback     []     @(iter, P, hist), called at each lsqnonlin
%                                   iteration (OutputFcn state 'iter', iteration >= 1)
%   Note: lsqnonlin performs maxIter+1 iterations before it reports the iteration
%   limit (31 for the default 30, as in the legacy baseline).
%
%   Outputs:
%     P    : solution returned by lsqnonlin
%     hist : struct
%              err        (nIter+1)x1  resnorm = sum(F.^2); err(1) at P0, err(k+1)
%                                      after iteration k (lsqnonlin's own count;
%                                      an iteration with an unsuccessful step
%                                      repeats the previous value)
%              stepNorm   nIterx1      2-norm of the change of x between OutputFcn calls
%              time       (nIter+1)x1  cumulative seconds
%              nIter      output.iterations
%              stopReason string: mapped exitflag (see below)
%              exitflag, message   raw lsqnonlin outputs
%   stopReason by exitflag: 1 'gradientTol' (first-order optimality),
%   2 'stepTol', 3 'functionTol', 4 'stepTol' (magnitude of search direction),
%   0 'maxIter', -1 'outputFcnStop', otherwise 'exitflag<n>'.

d = struct('maxIter', 30, 'functionTol', 1e-10, 'optimOptions', {{}}, ...
           'verbose', false, 'callback', []);
if nargin < 3, opts = struct(); end
o = parseOpts(opts, d, 'lsqnonlinBaseline');

display = 'off';
if o.verbose, display = 'iter'; end
t0 = tic;
hist = [];
xPrev = P0(:);

lo = optimoptions(@lsqnonlin, 'Display', display, 'SpecifyObjectiveGradient', true, ...
    'FunctionTolerance', o.functionTol, 'MaxIterations', o.maxIter, 'OutputFcn', @outfun);
if ~isempty(o.optimOptions)
    lo = optimoptions(lo, o.optimOptions{:});
end

[P, ~, ~, exitflag, output] = lsqnonlin(@(x) resFun(x, model), P0(:), [], [], lo);

hist.nIter = output.iterations;
hist.exitflag = exitflag;
hist.message = output.message;
switch exitflag
    case 1, hist.stopReason = 'gradientTol';
    case 2, hist.stopReason = 'stepTol';
    case 3, hist.stopReason = 'functionTol';
    case 4, hist.stopReason = 'stepTol';
    case 0, hist.stopReason = 'maxIter';
    case -1, hist.stopReason = 'outputFcnStop';
    otherwise, hist.stopReason = sprintf('exitflag%d', exitflag);
end
hist = orderfields(hist, {'err', 'stepNorm', 'time', 'nIter', 'stopReason', 'exitflag', 'message'});

    function stop = outfun(x, optimValues, state)
        stop = false;
        switch state
            case 'init'
                hist = histInit(optimValues.resnorm, toc(t0));
            case 'iter'
                if optimValues.iteration < 1, return, end
                hist = histPush(hist, optimValues.resnorm, norm(x(:) - xPrev), toc(t0));
                xPrev = x(:);
                reportIter('lsqnonlin', o, x(:), hist);
        end
    end
end

function [F, J] = resFun(P, model)
if nargout > 1
    [F, J] = edsm.residual(P, model);
else
    F = edsm.residual(P, model);
end
end
