function mesh = loadMesh(path)
%LOADMESH  Read a PLY triangle mesh.
%
%   mesh = edsm.loadMesh(path)
%
%   Input:
%     path : PLY file (ascii, binary_little_endian or binary_big_endian)
%   Output (struct):
%     V : n x 3 double vertex positions
%     F : m x 3 double face indices (1-based)
%     N : n x 3 double vertex normals, vertexNormal(triangulation(F, V))
%
%   Gives the same V and F as [F, V] = jwutils.plyread(path, 'tri'); no
%   dependency on jwutils. Polygons with more than 3 vertices are fan-
%   triangulated; the extra triangles are appended after the first m0 faces
%   (same order as plyread).

[V, F] = readPly(path);
mesh.V = V;
mesh.F = F;
mesh.N = vertexNormal(triangulation(F, V));
end
