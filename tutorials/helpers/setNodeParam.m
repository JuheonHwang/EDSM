function P = setNodeParam(P, n, i, A, t)
%SETNODEPARAM  Overwrite A_i and t_i of node i in a parameter vector.
%
%   P = setNodeParam(P, n, i, A, t)    n = number of nodes, A 3x3, t 3 elements
P(6 + 9 * (i - 1) + (1:9)) = A(:);
P(6 + 9 * n + 3 * (i - 1) + (1:3)) = t(:);
end
