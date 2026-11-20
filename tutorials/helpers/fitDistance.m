function d = fitDistance(Vdef, tarV)
%FITDISTANCE  Distance of each target vertex to its nearest deformed-source vertex.
%
%   d = fitDistance(Vdef, tarV)   % nTar x 1, same units as the mesh
idx = edsm.nearestIdx(Vdef, tarV);
d = sqrt(sum((Vdef(idx, :) - tarV).^2, 2));
end
