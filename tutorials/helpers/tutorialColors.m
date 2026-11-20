function c = tutorialColors()
%TUTORIALCOLORS  Colors shared by all tutorial figures (colorblind-safe, Okabe-Ito based).
%
%   Solver colors are used for the same solver in every figure.
hex = @(s) sscanf(s(2:end), '%2x')' / 255;
c.gd      = hex('#D55E00');   % gradient descent
c.gn      = hex('#CC79A7');   % Gauss-Newton
c.lm      = hex('#0072B2');   % Levenberg-Marquardt (direct, diag damping)
c.lmEye   = hex('#56B4E9');   % Levenberg-Marquardt, identity damping
c.lmCG    = hex('#009E73');   % LM + plain CG
c.lmPCG   = hex('#E69F00');   % LM + Jacobi-PCG
c.lsq     = hex('#555555');   % lsqnonlin
c.legacy  = hex('#999999');   % legacy-Jacobian runs (drawn dashed)
% Energy terms
c.fit     = hex('#0072B2');
c.rot     = hex('#D55E00');
c.smooth  = hex('#009E73');
c.lap     = hex('#CC79A7');
% Meshes
c.source  = hex('#3D7FC1');
c.target  = hex('#F28E2B');
c.neutral = [0.80 0.80 0.80];
c.text    = [0.15 0.15 0.15];
end
