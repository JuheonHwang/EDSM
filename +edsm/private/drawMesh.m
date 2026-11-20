function drawMesh(V, F, color)
%DRAWMESH  Add a patch (and lights) to the current axes; replaces jwutils.dispMesh.
%
%   drawMesh(V, F, color)
%     V     : n x 3 vertices
%     F     : m x 3 faces
%     color : [r g b] or [r g b alpha]

if numel(color) == 4
    alpha = color(4);
    color = color(1:3);
else
    color = color(1:3);
    alpha = 1;
end

FV.vertices = V;
FV.faces = F;
patch(FV, 'facecolor', color, 'edgecolor', 'none', 'vertexnormalsmode', 'auto', 'FaceAlpha', alpha);

camlight(180, 0);
camlight(0, 0);
lighting phong;
material dull;
axis vis3d;
axis equal;
end
