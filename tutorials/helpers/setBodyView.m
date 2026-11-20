function setBodyView(ax, bounds, varargin)
%SETBODYVIEW  Orthographic camera for the body meshes, head up, fitted to the axes.
%
%   setBodyView(ax, bounds)                 % bounds: 2x3 [min; max] of the region to show
%   setBodyView(ax, bounds, 'View', 'side', 'Pad', 0.04, 'Limits', fullBodyBounds)
%
%   The body stands along the data x axis (head at +x, feet and a floor slab at
%   -x). View 'front' (default) puts the camera at -y looking along +y, which shows
%   the front of the body; 'side' puts the camera at +z. The camera is fitted to
%   the pixel aspect of the axes, so create the axes with panelGrid or give them
%   an explicit Position. Two lights are added (headlight and a weaker fill).
p = inputParser;
addParameter(p, 'View', 'front');
addParameter(p, 'Pad', 0.04);
addParameter(p, 'Limits', []);   % 2x3 data limits (default: bounds enlarged); use the full-body bounds when bounds is a crop
parse(p, varargin{:});

c = mean(bounds, 1);
ext = diff(bounds, 1, 1);
switch p.Results.View
    case 'front'
        look = [0 1 0]; hExt = ext(3);
    case 'side'
        look = [0 0 -1]; hExt = ext(2);
    otherwise
        error('setBodyView:view', 'View must be ''front'' or ''side''.');
end
vExt = ext(1);
up = [1 0 0];
fig = ancestor(ax, 'figure');
oldU = ax.Units; ax.Units = 'normalized';
figPos = fig.Position;
asp = (ax.Position(3) * figPos(3)) / (ax.Position(4) * figPos(4));   % width / height
ax.Units = oldU;
% CameraViewAngle refers to the SHORTER side of the axes rectangle.
if asp < 1
    halfH = max(hExt / 2, vExt / 2 * asp) * (1 + p.Results.Pad);
else
    halfH = max(vExt / 2, hExt / 2 / asp) * (1 + p.Results.Pad);
end

lim = p.Results.Limits;
if isempty(lim), lim = bounds + [-1; 1] * 0.5 * max(ext); end
dist = 10 * max(diff(lim, 1, 1));
ax.Projection = 'orthographic';
ax.DataAspectRatio = [1 1 1];
ax.XLim = lim(:, 1)'; ax.YLim = lim(:, 2)'; ax.ZLim = lim(:, 3)';
ax.CameraTarget = c;
ax.CameraPosition = c - look * dist;
ax.CameraUpVector = up;
ax.CameraViewAngleMode = 'manual';
ax.CameraViewAngle = 2 * atan(halfH / dist) * 180 / pi;
ax.AmbientLightColor = [1 1 1];
ax.Visible = 'off';
delete(findall(ax, 'Type', 'light'));
light(ax, 'Style', 'infinite', 'Position', -look + 0.0 * up, 'Color', [1 1 1]);
light(ax, 'Style', 'infinite', 'Position', -look + 0.8 * up + [0 0 0.5], 'Color', [0.45 0.45 0.45]);
end
