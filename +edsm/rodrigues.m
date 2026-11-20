function [R, J] = rodrigues(r)
%RODRIGUES  Rotation vector to rotation matrix (pure MATLAB, replaces cv.Rodrigues).
%
%   [R, J] = edsm.rodrigues(r)
%
%   Input:
%     r : rotation vector, 3 elements (row or column). Axis = r/norm(r),
%         angle = norm(r), right-handed: R*x rotates x counter-clockwise about r.
%   Output:
%     R : 3x3 rotation matrix, R = cos(t) I + (1-cos(t)) a a' + sin(t) [a]_x
%     J : 3x9 Jacobian in OpenCV layout. Row i = d/dr_i, columns are the
%         elements of R in ROW-major order [R11 R12 R13 R21 R22 R23 R31 R32 R33].
%         (To get column-major R(:) order use J(:, [1 4 7 2 5 8 3 6 9]).)
%
%   Same convention and layout as cv.Rodrigues (vector-to-matrix branch). The
%   Jacobian is derived analytically (not copied from OpenCV) and checked
%   against finite differences in tests/test_core.m. For theta < eps the exact
%   derivative at r = 0 is returned, so the starting point r = 0 is valid.

r = double(r(:));
theta = norm(r);

I3 = eye(3);
if theta < eps
    % First-order expansion R = I + [r]_x; exact derivative at r = 0.
    R = I3;
    J = [0 0 0  0 0 -1  0 1 0;
         0 0 1  0 0  0 -1 0 0;
         0 -1 0 1 0  0  0 0 0];
    return
end

c = cos(theta);
s = sin(theta);
c1 = 1 - c;
itheta = 1 / theta;
a = r * itheta;                       % unit axis, 3x1

rrt = a * a';                         % 3x3
r_x = [0 -a(3) a(2); a(3) 0 -a(1); -a(2) a(1) 0];   % [a]_x
R = c * I3 + c1 * rrt + s * r_x;

if nargout > 1
    % Differentiate R = c I + c1 a a' + s [a]_x with a = r/theta:
    %   dtheta/dr_i = a_i,  da/dr_i = (e_i - a_i a)/theta,
    %   dc = -s a_i,  dc1 = s a_i,  ds = c a_i.
    J = zeros(3, 9);
    for i = 1:3
        ei = zeros(3, 1); ei(i) = 1;
        da = (ei - a(i) * a) * itheta;                           % 3x1
        dR = -s * a(i) * I3 ...
            + s * a(i) * rrt ...
            + c1 * (da * a' + a * da') ...
            + c * a(i) * r_x ...
            + s * [0 -da(3) da(2); da(3) 0 -da(1); -da(2) da(1) 0];   % dR/dr_i, 3x3
        J(i, :) = reshape(dR', 1, 9);                            % row-major flattening
    end
end
end
