function V = deform(P, model)
%DEFORM  Deformed source vertices for a parameter vector.
%
%   V = edsm.deform(P, model)
%
%   Inputs:
%     P     : (6 + 12n) parameter vector (see edsm.encodeParam)
%     model : struct from edsm.buildModel
%   Output:
%     V : nVerts x 3 deformed source vertices (vertices without valid ED
%         neighbors are returned unchanged, as in the originals).
%
%   Equivalent to deformED of the original scripts (vertices only; the
%   deformed normals are not computed).

[R0, T0, A0, t0] = edsm.decodeParam(P, model.nNodes);
d = deformCore(R0, T0, A0, t0, model);
V = d.V;
end
