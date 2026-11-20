function t = panelTitle(ax, str, varargin)
%PANELTITLE  Title above an axes whose Visible is off (mesh panels).
%
%   t = panelTitle(ax, 'text')    extra arguments are passed to annotation('textbox')
fig = ancestor(ax, 'figure');
oldU = ax.Units; ax.Units = 'normalized';
pos = ax.Position; ax.Units = oldU;
t = annotation(fig, 'textbox', [pos(1), pos(2) + pos(4), pos(3), 0.09], 'String', str, ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'FontSize', 11, 'Color', [0.15 0.15 0.15], 'Margin', 0, 'FitBoxToText', 'off', varargin{:});
end
