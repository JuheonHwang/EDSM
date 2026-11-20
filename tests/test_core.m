% TEST_CORE  Checks of the shared +edsm package.
%
%   Run from the repository root:
%       matlab -batch "setup; run('tests/test_core.m')"
%
%   (a) edsm.rodrigues Jacobian vs central finite differences
%   (b) analytic Jacobian of edsm.residual vs edsm.numericJacobian, per block
%   (c) edsm.deform(P0) equals the source vertices
%   (d) edsm.loadMesh equals jwutils.plyread (skipped if ED_Jacobian is absent)
%   plus small checks of edsm.nearestIdx, edsm.encodeParam/decodeParam and the
%   node sampling.
%
%   The corrected analytic Jacobian is asserted block by block over all rows and
%   columns. The legacy Jacobian (cfg.legacyJacobian = true) is only printed.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
rngBackup = rng;
rng(12345);
fprintf('==== edsm core tests ====\n');

%% (a) rodrigues
fprintf('\n[a] rodrigues Jacobian vs central differences (h = 1e-6)\n');
scales = [1e-9 1e-6 1e-3 0.1 0.5 1 2 3];
worst = 0;
for k = 1:numel(scales)
    r = randn(3, 1); r = r / norm(r) * scales(k);
    [R, J] = edsm.rodrigues(r);
    assert(norm(R' * R - eye(3)) < 1e-13 && abs(det(R) - 1) < 1e-13, 'R not a rotation');
    Jn = fdRodrigues(r);
    err = max(abs(J - Jn), [], 'all');
    worst = max(worst, err);
    fprintf('    |r| = %-8.1e  max|J - Jfd| = %.3g\n', scales(k), err);
    assert(err < 1e-8, 'rodrigues Jacobian mismatch at |r| = %g', scales(k));
end
[R0, J0] = edsm.rodrigues(zeros(3, 1));
Jn0 = fdRodrigues(zeros(3, 1));
fprintf('    r = 0: R == I: %d, max|J - Jfd| = %.3g\n', isequal(R0, eye(3)), max(abs(J0 - Jn0), [], 'all'));
assert(isequal(R0, eye(3)) && max(abs(J0 - Jn0), [], 'all') < 1e-8);
% encodeParam must invert rodrigues (also near pi)
testVecs = [randn(3, 4) * 0.7, [pi - 1e-3; 0; 0], [0; 0; -(pi - 1e-3)], zeros(3, 1)];
for k = 1:size(testVecs, 2)
    rv = testVecs(:, k);
    Pp = edsm.encodeParam(edsm.rodrigues(rv), [0 0 0], eye(3), [0; 0; 0]);
    assert(max(abs(Pp(1:3) - rv)) < 1e-7, 'encodeParam does not invert rodrigues');
end
fprintf('    worst rodrigues error: %.3g   encodeParam inverts rodrigues: OK\n', worst);

%% model
srcPath = fullfile(repoRoot, 'data', 'simple_poisson_mesh_020.ply');
tarPath = fullfile(repoRoot, 'data', 'simple_poisson_mesh_022.ply');
src = edsm.loadMesh(srcPath);
tar = edsm.loadMesh(tarPath);
model = edsm.buildModel(src, tar);
nNodes = model.nNodes;
P0 = edsm.initParam(model);
fprintf('\nmodel: %d source vertices, %d target vertices, %d ED nodes, %d invalid-neighbor vertices\n', ...
    model.nVerts, size(tar.V, 1), nNodes, nnz(any(isnan(model.idxNeighbor), 1)));

% node sampling reproduces the original statements
rng(1, 'twister'); expectNodes = randperm(model.nVerts, nNodes);
assert(isequal(model.nodeIdx, expectNodes), 'node sampling differs');
rng(12345);   % (buildModel restores the generator; keep the test stream deterministic)
assert(isequal(size(model.idxNeighbor), [4 model.nVerts]) && isequal(size(model.idxNodeNeighbor), [6 nNodes]));
assert(all(model.idxNodeNeighbor(1, :) == 1:nNodes), 'geodesic: node itself must be the first node neighbor');

%% (b) analytic vs numeric Jacobian (corrected mode, every row and column asserted)
assert(~model.cfg.legacyJacobian, 'default config must select the corrected Jacobian');

Pbase = P0;
Prand = P0;
Prand(1:3) = 0.3 * randn(3, 1);
Prand(4:6) = 0.5 * randn(3, 1);
Arand = repmat(eye(3), 1, 1, nNodes) + 0.2 * randn(3, 3, nNodes);
Prand(7:(6 + 9 * nNodes)) = Arand(:);
Prand((7 + 9 * nNodes):end) = 0.5 * randn(3 * nNodes, 1);

fprintf('\n[b] corrected analytic Jacobian vs central differences, no rows masked\n');
checkJacobian(model, Pbase, 'P0 (geodesic)', true);
checkJacobian(model, Prand, 'Prand (geodesic)', true);

% Euclidean distances (distanceOnMesh = false)
cfgE = edsm.defaultConfig(); cfgE.distanceOnMesh = false;
modelE = edsm.buildModel(src, tar, cfgE);
assert(isequal(size(modelE.idxNodeNeighbor), size(model.idxNodeNeighbor)) && ...
    all(modelE.idxNodeNeighbor(1, :) == 1:nNodes), 'Euclidean node neighbors: self first, nNodeNeighbor entries');
assert(isequal(size(modelE.idxNeighbor), size(model.idxNeighbor)));
checkJacobian(modelE, Prand, 'Prand (Euclidean)', true);   % same node count -> same P layout

% The 9 invalid vertices of the sample mesh form a connected cluster that is
% isolated from the valid vertices in the Laplacian, so the lap x T block is zero
% there. Flag extra vertices (among them interior ones with valid neighbors,
% matched by the fit term) as invalid to exercise the general case.
modelX = model;
extra = round(linspace(50, model.nVerts - 50, 40));
modelX.idxNeighbor(:, extra) = NaN;
[~, JX, infoX] = edsm.residual(Prand, modelX);
lapRows = infoX.rows.lap(1):infoX.rows.lap(2);
assert(nnz(JX(lapRows, 4:6)) > 0, 'expected nonzero lap x T entries next to invalid vertices');
checkJacobian(modelX, Prand, 'Prand (geodesic, +40 extra invalid vertices)', true);

%% (b2) legacy Jacobian: documents the known errors (printed, NOT asserted)
fprintf('\n[b2] legacy Jacobian (cfg.legacyJacobian = true): known errors, not asserted\n');
modelL = model; modelL.cfg.legacyJacobian = true;
[Fc, ~] = edsm.residual(Prand, model);
[Fl, ~] = edsm.residual(Prand, modelL);
assert(isequal(Fc, Fl), 'F must not depend on legacyJacobian');
fprintf('    F identical in both modes: %d\n', isequal(Fc, Fl));
checkJacobian(modelL, P0, 'P0 (legacy)', false);
checkJacobian(modelL, Prand, 'Prand (legacy)', false);

%% (c) deform
fprintf('\n[c] deform(P0) vs source vertices\n');
Vd = edsm.deform(P0, model);
dev = max(abs(Vd - src.V), [], 'all');
extent = max(max(src.V) - min(src.V));
fprintf('    max abs deviation = %.3g (mesh extent %.3g)\n', dev, extent);
assert(dev < 1e-9 * extent, 'deform(P0) differs from the source');
[~, ~, info0] = edsm.residual(P0, model);
assert(isequal(Vd, info0.V), 'deform and residual disagree on the deformed vertices');

%% small utility checks
fprintf('\n[extra] nearestIdx, decodeParam/encodeParam\n');
X = rand(700, 3); Y = rand(300, 3);
D2 = sum((permute(Y, [1 3 2]) - permute(X, [3 1 2])).^2, 3);
[~, ref] = min(D2, [], 2);
assert(isequal(edsm.nearestIdx(X, Y), ref), 'nearestIdx differs from brute force');
Xt = [0 0 0; 1 0 0; 1 0 0; 0 0 0]; assert(isequal(edsm.nearestIdx(Xt, [1 0 0; 0 0 0]), [2; 1]), 'tie rule');
[Rr, Tr, Ar, tr] = edsm.decodeParam(Prand, nNodes);
Pr2 = edsm.encodeParam(Rr, Tr, Ar, tr);
fprintf('    encode(decode(P)) max abs diff = %.3g\n', max(abs(Pr2 - Prand)));
assert(max(abs(Pr2 - Prand)) < 1e-12);

%% (d) loadMesh vs jwutils.plyread
fprintf('\n[d] loadMesh vs jwutils.plyread\n');
jwRoot = '';
for cand = {fullfile(fileparts(repoRoot), 'ED_Jacobian'), '/Users/juheon/Desktop/works/ED_Jacobian'}
    if isfolder(fullfile(cand{1}, '+jwutils')), jwRoot = cand{1}; break; end
end
if isempty(jwRoot)
    fprintf('    SKIPPED: ED_Jacobian/+jwutils not found.\n');
else
    addpath(jwRoot);
    for pth = {srcPath, tarPath}
        [Fj, Vj] = jwutils.plyread(pth{1}, 'tri');
        m = edsm.loadMesh(pth{1});
        eqV = isequal(m.V, Vj); eqF = isequal(m.F, Fj);
        [~, nm] = fileparts(pth{1});
        fprintf('    %s: V equal %d, F equal %d (V %s, F %s)\n', nm, eqV, eqF, mat2str(size(m.V)), mat2str(size(m.F)));
        assert(eqV && eqF, 'loadMesh differs from jwutils.plyread');
        assert(isequal(m.N, vertexNormal(triangulation(Fj, Vj))), 'normals differ');
    end
    rmpath(jwRoot);
end

rng(rngBackup);
fprintf('\n==== all asserted checks passed ====\n');

function Jn = fdRodrigues(r)
% Central finite differences, OpenCV layout (3x9, row i = d/dr_i, row-major R).
h = 1e-6;
Jn = zeros(3, 9);
for i = 1:3
    e = zeros(3, 1); e(i) = h;
    d = (edsm.rodrigues(r + e) - edsm.rodrigues(r - e)) / (2 * h);
    Jn(i, :) = reshape(d', 1, 9);
end
end

function checkJacobian(model, P, label, doAssert)
% Compare edsm.residual's analytic J with edsm.numericJacobian over all rows and
% all columns; print the max abs error per (row block x column block).
nNodes = model.nNodes;
colBlocks = {'r', 1:3; 'T', 4:6; 'A', 7:(6 + 9 * nNodes); 't', (7 + 9 * nNodes):numel(P)};
[~, J, info] = edsm.residual(P, model);
tic; Jn = edsm.numericJacobian(P, model); tNum = toc;
Jf = full(J);
fprintf('  -- %s: size(J) = %s, %d invalid-neighbor vertices, numeric Jacobian %.1f s, max abs err overall %.3g\n', ...
    label, mat2str(size(J)), nnz(any(isnan(model.idxNeighbor), 1)), tNum, max(abs(Jf - Jn), [], 'all'));
names = fieldnames(info.rows);
fprintf('    %-12s %12s %12s   %s\n', 'block', 'max abs err', 'rel err', 'status');
for i = 1:numel(names)
    rr = info.rows.(names{i})(1):info.rows.(names{i})(2);
    for c = 1:size(colBlocks, 1)
        cc = colBlocks{c, 2};
        scale = max(abs(Jn(rr, cc)), [], 'all');
        maxErr = max(abs(Jf(rr, cc) - Jn(rr, cc)), [], 'all');
        ok = maxErr <= 1e-6 * max(1, scale);
        if ok, status = 'ok'; elseif doAssert, status = 'FAIL'; else, status = 'differs (legacy)'; end
        fprintf('    %-12s %12.3g %12.3g   %s\n', sprintf('%s x %s', names{i}, colBlocks{c, 1}), ...
            maxErr, maxErr / max(scale, eps), status);
        if doAssert
            assert(ok, 'Jacobian block %s x %s mismatch at %s', names{i}, colBlocks{c, 1}, label);
        end
    end
end
end
