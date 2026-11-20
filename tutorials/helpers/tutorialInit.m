function ctx = tutorialInit(scriptName)
%TUTORIALINIT  Common start-up of every tutorial script.
%
%   ctx = tutorialInit('t00_quickstart')
%
%   Puts the repository root and tutorials/helpers on the path, creates
%   tutorials/output and docs/img, starts a diary that records the console
%   output in tutorials/output/<scriptName>.log, and returns a context struct
%     name, repoRoot, tutDir, outDir, imgDir, logFile, interactive, tStart.
%   Call tutorialDone(ctx) at the end of the script.
%   Figures are invisible when MATLAB runs without a desktop (-batch).

helpersDir = fileparts(mfilename('fullpath'));
tutDir = fileparts(helpersDir);
repoRoot = fileparts(tutDir);
addpath(helpersDir);
addpath(repoRoot);

ctx.name = scriptName;
ctx.repoRoot = repoRoot;
ctx.tutDir = tutDir;
ctx.outDir = fullfile(tutDir, 'output');
ctx.imgDir = fullfile(repoRoot, 'docs', 'img');
ctx.logFile = fullfile(ctx.outDir, [scriptName '.log']);
ctx.interactive = usejava('desktop');
if ~exist(ctx.outDir, 'dir'), mkdir(ctx.outDir); end
if ~exist(ctx.imgDir, 'dir'), mkdir(ctx.imgDir); end

diary off
if exist(ctx.logFile, 'file'), delete(ctx.logFile); end
diary(ctx.logFile);
diary on
ctx.tStart = tic;
ctx.figures = {};
fprintf('=== %s  (%s) ===\n', scriptName, char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf('MATLAB %s\n\n', version('-release'));
end
