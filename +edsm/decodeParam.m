function [R, T, A, t, J_R] = decodeParam(P, nNodes)
%DECODEPARAM  Unpack the ED parameter vector.
%
%   [R, T, A, t, J_R] = edsm.decodeParam(P, nNodes)
%
%   Inputs:
%     P      : (6 + 12*nNodes) vector, layout [r(3); T(3); A(:) (9n); t(:) (3n)]
%     nNodes : number of ED nodes n
%   Outputs:
%     R   : 3x3 rotation matrix, R = edsm.rodrigues(P(1:3))
%     T   : 1x3 translation
%     A   : 3x3xn affine matrices (A(:,:,i) = reshape(P(6+9*(i-1)+(1:9)), 3, 3))
%     t   : 3xn node translations
%     J_R : 3x9 Jacobian of R w.r.t. r, OpenCV layout (see edsm.rodrigues)

P = P(:);
[R, J_R] = edsm.rodrigues(P(1:3)');
T = P(4:6)';
nA = 9 * nNodes;
A = P(7:(7 + nA - 1));
t = P((7 + nA):end);
A = reshape(A, [3, 3, nNodes]);
t = reshape(t, [3, nNodes]);
end
