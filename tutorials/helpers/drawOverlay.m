function drawOverlay(ax, srcV, srcF, tarV, tarF)
%DRAWOVERLAY  Source (blue) and target (orange) surfaces in one axes.
%
%   Both surfaces are opaque. The depth buffer decides which one is visible, so a
%   patchwork of both colors means the surfaces coincide (they interpenetrate
%   at the scale of the mesh noise) and a large single-color region means the
%   surfaces are apart there; the surface in front shows its color.
col = tutorialColors();
drawSurface(ax, tarV, tarF, 'Color', col.target);
drawSurface(ax, srcV, srcF, 'Color', col.source);
end
