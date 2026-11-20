function h = drawSurface(ax, V, F, varargin)
%DRAWSURFACE  Add a lit triangle mesh to an axes.
%
%   h = drawSurface(ax, V, F)                                % neutral gray
%   h = drawSurface(ax, V, F, 'Color', rgb, 'Alpha', 0.5)
%   h = drawSurface(ax, V, F, 'CData', perVertexValues)      % colormap/clim of the axes
%
%   Lights are added by setBodyView (call it after drawing).
p = inputParser;
addParameter(p, 'Color', [0.8 0.8 0.8]);
addParameter(p, 'Alpha', 1);
addParameter(p, 'CData', []);
parse(p, varargin{:});
common = {'EdgeColor', 'none', 'FaceAlpha', p.Results.Alpha, 'FaceLighting', 'gouraud', ...
    'AmbientStrength', 0.65, 'DiffuseStrength', 0.45, 'SpecularStrength', 0.05, 'BackFaceLighting', 'lit'};
if isempty(p.Results.CData)
    h = patch(ax, 'Faces', F, 'Vertices', V, 'FaceColor', p.Results.Color, common{:});
else
    h = patch(ax, 'Faces', F, 'Vertices', V, 'FaceVertexCData', p.Results.CData(:), ...
        'FaceColor', 'interp', common{:});
end
hold(ax, 'on');
end
