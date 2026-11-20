function o = parseOpts(opts, defaults, caller)
%PARSEOPTS  Merge a user option struct with defaults (shared solver helper).
%
%   o = parseOpts(opts, defaults, caller)
%
%   Missing or empty fields of OPTS take the value from DEFAULTS. A field of
%   OPTS that is not a field of DEFAULTS raises edsm:solvers:unknownOption
%   (guards against typos). CALLER is the solver name used in the message.

if nargin < 1 || isempty(opts), opts = struct(); end
if ~isstruct(opts)
    error('edsm:solvers:badOptions', '%s: opts must be a struct.', caller);
end
o = defaults;
names = fieldnames(opts);
for i = 1:numel(names)
    nm = names{i};
    if ~isfield(defaults, nm)
        error('edsm:solvers:unknownOption', '%s: unknown option "%s".', caller, nm);
    end
    if ~isempty(opts.(nm))
        o.(nm) = opts.(nm);
    end
end
end
