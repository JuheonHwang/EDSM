function [R, J] = Rodrigues(r)
% Shim for mexopencv cv.Rodrigues (rotation vector -> matrix only).
% J is 3x9: row i = dR/dr_i, columns = elements of R in ROW-major order.
r = double(r(:));
th = norm(r);
K = [0 -r(3) r(2); r(3) 0 -r(1); -r(2) r(1) 0];
if th < 0.05
    a  = 1 - th^2/6 + th^4/120 - th^6/5040;
    b  = 0.5 - th^2/24 + th^4/720 - th^6/40320;
    da = -1/3 + th^2/30 - th^4/840;     % a'(th)/th
    db = -1/12 + th^2/180 - th^4/6720;  % b'(th)/th
else
    a  = sin(th)/th;
    b  = (1 - cos(th))/th^2;
    da = (th*cos(th) - sin(th))/th^3;
    db = (th*sin(th) - 2*(1 - cos(th)))/th^4;
end
R = eye(3) + a*K + b*(K*K);
if nargout > 1
    J = zeros(3, 9);
    for i = 1:3
        e = zeros(3,1); e(i) = 1;
        E = [0 -e(3) e(2); e(3) 0 -e(1); -e(2) e(1) 0];
        dR = (da*r(i))*K + a*E + (db*r(i))*(K*K) + b*(E*K + K*E);
        J(i, :) = reshape(dR.', 1, 9);   % row-major
    end
end
end
