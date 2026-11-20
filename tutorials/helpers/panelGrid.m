function axs = panelGrid(fig, nr, nc, varargin)
%PANELGRID  Axes on a regular grid with explicit margins (so the aspect of each panel is known).
%
%   axs = panelGrid(fig, nr, nc)               % nr x nc array of axes
%   axs = panelGrid(fig, nr, nc, 'Margins', [left right bottom top], 'Gap', [gx gy])
%
%   Margins and gaps are in normalized figure units. Mesh panels need this because
%   setBodyView fits the camera to the pixel size of the axes.
p = inputParser;
addParameter(p, 'Margins', [0.02 0.02 0.02 0.08]);
addParameter(p, 'Gap', [0.01 0.02]);
parse(p, varargin{:});
m = p.Results.Margins; g = p.Results.Gap;
w = (1 - m(1) - m(2) - (nc - 1) * g(1)) / nc;
h = (1 - m(3) - m(4) - (nr - 1) * g(2)) / nr;
axs = gobjects(nr, nc);
for i = 1:nr
    for j = 1:nc
        x = m(1) + (j - 1) * (w + g(1));
        y = 1 - m(4) - i * h - (i - 1) * g(2);
        axs(i, j) = axes(fig, 'Position', [x y w h]);
    end
end
end
