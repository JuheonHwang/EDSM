function lims = showMeshes(V1, F1, V2, F2, rot1, rot2, lims, varargin)
%SHOWMESHES  Overlay two meshes in a new figure (replaces the local dispMeshes).
%
%   lims = edsm.showMeshes(V1, F1, V2, F2)
%   lims = edsm.showMeshes(V1, F1, V2, F2, rot1, rot2, lims)
%   lims = edsm.showMeshes(..., lims, 'Visible', 'off')
%
%   Inputs:
%     V1, F1 : first mesh (n1 x 3, m1 x 3), drawn red   [0.5 0.2 0.2], alpha 0.3
%     V2, F2 : second mesh (n2 x 3, m2 x 3), drawn green [0.2 0.5 0.2], alpha 0.3
%     rot1   : camera orbit (deg) about the data z axis (default -90)
%     rot2   : camera orbit (deg) about the data x axis (default 0)
%     lims   : [] to record the axis limits of this figure, or a 3x2
%              [xlim; ylim; zlim] to reuse limits from an earlier figure
%   Output:
%     lims : 3x2 [xlim; ylim; zlim] (the input lims if it was non-empty)
%
%   Pass lims = [] on the first call and the returned value on later calls so
%   that all iterations are shown in the same frame.

if nargin < 5 || isempty(rot1), rot1 = -90; end
if nargin < 6 || isempty(rot2), rot2 = 0; end
if nargin < 7, lims = []; end
visible = 'on';
if ~isempty(varargin)
    assert(numel(varargin) == 2 && strcmpi(varargin{1}, 'Visible'), 'Unknown option.');
    visible = varargin{2};
end

figure('Visible', visible);
drawMesh(V1, F1, [0.5 0.2 0.2 0.3]);
drawMesh(V2, F2, [0.2 0.5 0.2 0.3]);
camorbit(rot1, 0, 'data', [0 0 1]);
camorbit(rot2, 0, 'data', [1 0 0]);
axis off;
if isempty(lims)
    lims = [xlim; ylim; zlim];
else
    xlim(lims(1, :)); ylim(lims(2, :)); zlim(lims(3, :));
end
end
