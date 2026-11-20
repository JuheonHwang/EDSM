function P0 = initParam(model)
%INITPARAM  Identity starting point of the registration.
%
%   P0 = edsm.initParam(model)
%
%   Input:
%     model : struct from edsm.buildModel
%   Output:
%     P0 : (6 + 12n) x 1 column [r(3); T(3); A(:) (9n); t(:) (3n)] with
%          R = I, T = 0, A_i = I, t_i = 0 (n = model.nNodes).

n = model.nNodes;
R0 = eye(3);
T0 = [0, 0, 0];
A0 = repmat(reshape(eye(3), [3 3 1]), [1 1 n]);
t0 = zeros(3, n);
P0 = edsm.encodeParam(R0, T0, A0, t0);
end
