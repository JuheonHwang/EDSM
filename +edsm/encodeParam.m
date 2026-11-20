function P = encodeParam(R, T, A, t)
%ENCODEPARAM  Pack ED parameters into one column vector.
%
%   P = edsm.encodeParam(R, T, A, t)
%
%   Inputs:
%     R : 3x3 global rotation matrix
%     T : 3 elements, global translation (row or column)
%     A : 3x3xn per-node affine matrices
%     t : 3xn per-node translations
%   Output:
%     P : (6 + 12n) x 1 column, layout [r(3); T(3); A(:) (9n); t(:) (3n)]
%         where r is the rotation vector of R in the edsm.rodrigues convention
%         (R = edsm.rodrigues(r)).
%
%   Note: the original scripts used MATLAB's rotationMatrixToVector, whose
%   convention is the transpose of cv.Rodrigues. The two agree only for R = I.
%   Here the inverse is consistent with edsm.rodrigues.

n = size(A, 3);
assert(size(t, 2) == n);
P = [rotationMatrixToVectorLocal(R), T(:)', A(:)', t(:)']';
end

function r = rotationMatrixToVectorLocal(R)
% Inverse of edsm.rodrigues: returns a 1x3 rotation vector.
w = [R(3,2) - R(2,3); R(1,3) - R(3,1); R(2,1) - R(1,2)] / 2;   % = sin(theta) * axis
cosT = (trace(R) - 1) / 2;
sinT = norm(w);
theta = atan2(sinT, cosT);
if sinT > 1e-8
    r = theta * w / sinT;
elseif cosT > 0
    r = w;                      % theta ~ 0, first order
else
    % theta ~ pi: axis from the symmetric part (R + I)/2 = a a'.
    B = (R + eye(3)) / 2;
    [~, k] = max(diag(B));
    a = B(:, k) / sqrt(B(k, k));
    if dot(a, w) < 0            % sign from the (tiny) skew part when available
        a = -a;
    end
    r = pi * a;
end
r = r(:)';
end
