function [names, e, rows] = termEnergies(F, info)
%TERMENERGIES  Contribution of each energy term to sum(F.^2).
%
%   [names, e, rows] = termEnergies(F, info)
%
%   names : {'Efit','Erot','Esmooth','Elap'}  (Erot = rot1 + rot2 rows)
%   e     : 1x4 sums of squares of the corresponding rows of F (weights included)
%   rows  : 4x2 [first last] row ranges (Erot covers rot1 and rot2, which are adjacent)
names = {'Efit', 'Erot', 'Esmooth', 'Elap'};
r = info.rows;
rows = [r.fit; r.rot1(1), r.rot2(2); r.smooth; r.lap];
e = zeros(1, 4);
for k = 1:4
    e(k) = sum(F(rows(k, 1):rows(k, 2)).^2);
end
end
