function fig = showMesh(V, F, rot1, rot2, varargin)
%SHOWMESH  Show one mesh in a new figure (replaces the local dispMesh).
%
%   fig = edsm.showMesh(V, F)
%   fig = edsm.showMesh(V, F, rot1, rot2)
%   fig = edsm.showMesh(V, F, rot1, rot2, 'Visible', 'off')
%
%   Inputs:
%     V    : n x 3 vertices
%     F    : m x 3 faces (1-based)
%     rot1 : camera orbit in degrees about the data z axis (default -90)
%     rot2 : camera orbit in degrees about the data x axis (default 0)
%     'Visible' : 'on' (default) or 'off' for the figure
%   Output:
%     fig : figure handle
%
%   Same look as the originals: opaque gray [0.8 0.8 0.8], no edges, two
%   camera lights, phong lighting, dull material, axis off.

if nargin < 3 || isempty(rot1), rot1 = -90; end
if nargin < 4 || isempty(rot2), rot2 = 0; end
visible = parseVisible(varargin);

fig = figure('Visible', visible);
drawMesh(V, F, [0.8 0.8 0.8 1.0]);
camorbit(rot1, 0, 'data', [0 0 1]);
camorbit(rot2, 0, 'data', [1 0 0]);
axis off;
end

function visible = parseVisible(args)
visible = 'on';
if ~isempty(args)
    assert(numel(args) == 2 && strcmpi(args{1}, 'Visible'), 'Unknown option.');
    visible = args{2};
end
end
