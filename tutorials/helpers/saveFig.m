function file = saveFig(ctx, fig, name)
%SAVEFIG  Save a figure as docs/img/<script>_<name>.png at 150 dpi.
file = fullfile(ctx.imgDir, sprintf('%s_%s.png', ctx.name, name));
for ax = findall(fig, 'Type', 'axes')'
    try, ax.XMinorGrid = 'off'; ax.YMinorGrid = 'off'; catch, end %#ok<NOCOM>
end
drawnow;
print(fig, file, '-dpng', '-r150');
info = imfinfo(file);
fprintf('[figure] %s  (%d x %d px)\n', relPath(ctx, file), info.Width, info.Height);
end

function s = relPath(ctx, file)
% Repo-relative path with forward slashes (logs are committed, no home dirs).
s = strrep(file, [ctx.repoRoot filesep], '');
s = strrep(s, filesep, '/');
end
