% SETUP  Add the EDSM repository root to the MATLAB path and check requirements.
%
%   Run once per MATLAB session from anywhere:   run('<repo>/setup.m')   or   setup
%
%   Required: MATLAB R2020b or newer (pagemtimes, isMATLABReleaseOlderThan) and
%   core functions graph / distances / mink / triangulation.
%   The +edsm package needs NO toolbox. Optional toolboxes are only reported:
%     Optimization Toolbox      (lsqnonlin-based solver scripts)
%     Parallel Computing Toolbox (parfor in edsm.numericJacobian, solver scripts)

edsmRoot = fileparts(mfilename('fullpath'));
addpath(edsmRoot);
fprintf('EDSM: added %s to the path.\n', edsmRoot);

edsmOk = true;

% ---- MATLAB release -------------------------------------------------------
if exist('isMATLABReleaseOlderThan', 'file') ~= 2 || isMATLABReleaseOlderThan('R2020b')
    fprintf('EDSM: MISSING - MATLAB R2020b or newer is required (found %s).\n', version('-release'));
    edsmOk = false;
else
    fprintf('EDSM: MATLAB %s OK.\n', version('-release'));
end

% ---- required core functions ---------------------------------------------
edsmFuncs = {'pagemtimes', 'mink', 'graph', 'triangulation', 'laplacian'};
for edsmK = 1:numel(edsmFuncs)
    if isempty(which(edsmFuncs{edsmK}))
        fprintf('EDSM: MISSING - function %s is not available.\n', edsmFuncs{edsmK});
        edsmOk = false;
    end
end

% ---- optional toolboxes ---------------------------------------------------
edsmInstalled = {ver().Name};
edsmOptional = {'Optimization Toolbox', 'Parallel Computing Toolbox'};
for edsmK = 1:numel(edsmOptional)
    if any(strcmp(edsmInstalled, edsmOptional{edsmK}))
        fprintf('EDSM: optional %s found.\n', edsmOptional{edsmK});
    else
        fprintf('EDSM: optional %s NOT found (only needed by some solvers / parfor).\n', edsmOptional{edsmK});
    end
end

if edsmOk
    fprintf('EDSM: setup complete.\n');
else
    fprintf('EDSM: setup finished with missing requirements (see above).\n');
end
clear edsmRoot edsmOk edsmFuncs edsmK edsmInstalled edsmOptional
